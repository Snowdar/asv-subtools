
# Copyright 2016 Vijayaditya Peddinti.
#           2016 Vimal Manohar
#           2017 Johns Hopkins University (author: Daniel Povey)
# Apache 2.0.

""" This is a module with methods which will be used by scripts for training of
deep neural network acoustic model and raw model (i.e., generic neural
network without transition model) with frame-level objectives.
"""

import glob
import logging
import math
import os
import random
import time

import libs.common as common_lib
import libs.nnet3.train.common as common_train_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def train_new_models(dir, iter, srand, num_jobs,
                     num_archives_processed, num_archives,
                     raw_model_string, egs_dir,
                     momentum, max_param_change,
                     shuffle_buffer_size, minibatch_size_str,
                     image_augmentation_opts,
                     run_opts, frames_per_eg=-1,
                     min_deriv_time=None, max_deriv_time_relative=None,
                     use_egs=False,
                     backstitch_training_scale=0.0, backstitch_training_interval=1):
    """ Called from train_one_iteration(), this model does one iteration of
    training with 'num_jobs' jobs, and writes files like
    exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw

    We cannot easily use a single parallel SGE job to do the main training,
    because the computation of which archive and which --frame option
    to use for each job is a little complex, so we spawn each one separately.
    this is no longer true for RNNs as we use do not use the --frame option
    but we use the same script for consistency with FF-DNN code

    Selected args:
        frames_per_eg:
            The frames_per_eg, in the context of (non-chain) nnet3 training,
            is normally the number of output (supervised) frames in each training
            example.  However, the frames_per_eg argument to this function should
            only be set to that number (greater than zero) if you intend to
            train on a single frame of each example, on each minibatch.  If you
            provide this argument >0, then for each training job a different
            frame from the dumped example is selected to train on, based on
            the option --frame=n to nnet3-copy-egs.
            If you leave frames_per_eg at its default value (-1), then the
            entire sequence of frames is used for supervision.  This is suitable
            for RNN training, where it helps to amortize the cost of computing
            the activations for the frames of context needed for the recurrence.
        use_egs : True, if different examples used to train multiple
            tasks or outputs, e.g.multilingual training.  multilingual egs can
            be generated using get_egs.sh and
            subtools/kaldi/steps_multitask/nnet3/multilingual/allocate_multilingual_examples.py, those
            are the top-level scripts.
    """

    chunk_level_training = False if frames_per_eg > 0 else True

    deriv_time_opts = []
    if min_deriv_time is not None:
        deriv_time_opts.append("--optimization.min-deriv-time={0}".format(
                           min_deriv_time))
    if max_deriv_time_relative is not None:
        deriv_time_opts.append("--optimization.max-deriv-time-relative={0}".format(
                           max_deriv_time_relative))

    threads = []

    # the GPU timing info is only printed if we use the --verbose=1 flag; this
    # slows down the computation slightly, so don't accumulate it on every
    # iteration.  Don't do it on iteration 0 either, because we use a smaller
    # than normal minibatch size, and people may get confused thinking it's
    # slower for iteration 0 because of the verbose option.
    verbose_opt = ("--verbose=1" if iter % 20 == 0 and iter > 0 else "")

    for job in range(1, num_jobs+1):
        # k is a zero-based index that we will derive the other indexes from.
        k = num_archives_processed + job - 1

        # work out the 1-based archive index.
        archive_index = (k % num_archives) + 1

        if not chunk_level_training:
            frame = (k / num_archives + archive_index) % frames_per_eg

        cache_io_opts = (("--read-cache={dir}/cache.{iter}".format(dir=dir,
                                                                  iter=iter)
                          if iter > 0 else "") +
                         (" --write-cache={0}/cache.{1}".format(dir, iter + 1)
                          if job == 1 else ""))

        if image_augmentation_opts:
            image_augmentation_cmd = (
                'nnet3-egs-augment-image --srand={srand} {aug_opts} ark:- ark:- |'.format(
                    srand=k+srand,
                    aug_opts=image_augmentation_opts))
        else:
            image_augmentation_cmd = ''


        multitask_egs_opts = common_train_lib.get_egs_opts(
            egs_dir,
            egs_prefix="egs.",
            archive_index=archive_index,
            use_egs=use_egs)

        scp_or_ark = "scp" if use_egs else "ark"

        egs_rspecifier = (
            """ark,bg:nnet3-copy-egs {frame_opts} {multitask_egs_opts} \
            {scp_or_ark}:{egs_dir}/egs.{archive_index}.{scp_or_ark} ark:- | \
            nnet3-shuffle-egs --buffer-size={shuffle_buffer_size} \
            --srand={srand} ark:- ark:- | {aug_cmd} \
            nnet3-merge-egs --minibatch-size={minibatch_size} ark:- ark:- |""".format(
                frame_opts=("" if chunk_level_training
                            else "--frame={0}".format(frame)),
                egs_dir=egs_dir, archive_index=archive_index,
                shuffle_buffer_size=shuffle_buffer_size,
                minibatch_size=minibatch_size_str,
                aug_cmd=image_augmentation_cmd,
                srand=iter+srand,
                scp_or_ark=scp_or_ark,
                multitask_egs_opts=multitask_egs_opts))

        # note: the thread waits on that process's completion.
        thread = common_lib.background_command(
            """{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
                    nnet3-train {parallel_train_opts} {cache_io_opts} \
                     {verbose_opt} --print-interval=10 \
                    --momentum={momentum} \
                    --max-param-change={max_param_change} \
                    --backstitch-training-scale={backstitch_training_scale} \
                    --l2-regularize-factor={l2_regularize_factor} \
                    --backstitch-training-interval={backstitch_training_interval} \
                    --srand={srand} \
                    {deriv_time_opts} "{raw_model}" "{egs_rspecifier}" \
                    {dir}/{next_iter}.{job}.raw""".format(
                command=run_opts.command,
                train_queue_opt=run_opts.train_queue_opt,
                dir=dir, iter=iter,
                next_iter=iter + 1, srand=iter + srand,
                job=job,
                parallel_train_opts=run_opts.parallel_train_opts,
                cache_io_opts=cache_io_opts,
                verbose_opt=verbose_opt,
                momentum=momentum, max_param_change=max_param_change,
                l2_regularize_factor=1.0/num_jobs,
                backstitch_training_scale=backstitch_training_scale,
                backstitch_training_interval=backstitch_training_interval,
                deriv_time_opts=" ".join(deriv_time_opts),
                raw_model=raw_model_string,
                egs_rspecifier=egs_rspecifier),
            require_zero_status=True)

        threads.append(thread)
        print("sleep 7s...")
        time.sleep(7)
    for thread in threads:
        thread.join()


def train_cvector_new_models(dir, iter, srand, num_jobs,
                     num_archives_processed, num_archives,
                     raw_model_string, 
                     am_output_name, am_weight, am_egs_dir,
                     xvec_output_name, xvec_weight, xvec_egs_dir,
                     momentum, max_param_change,
                     shuffle_buffer_size, minibatch_size_str,
                     image_augmentation_opts,
                     run_opts, 
                     am_frames_per_eg=-1,
                     min_deriv_time=None, max_deriv_time_relative=None,
                     use_egs=False,
                     backstitch_training_scale=0.0, backstitch_training_interval=1):
    """ Called from train_one_iteration(), this model does one iteration of
    training with 'num_jobs' jobs, and writes files like
    exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw

    We cannot easily use a single parallel SGE job to do the main training,
    because the computation of which archive and which --frame option
    to use for each job is a little complex, so we spawn each one separately.
    this is no longer true for RNNs as we use do not use the --frame option
    but we use the same script for consistency with FF-DNN code

    Selected args:
        am_frames_per_eg:
            The frames_per_eg, in the context of (non-chain) nnet3 training,
            is normally the number of output (supervised) frames in each training
            example.  However, the frames_per_eg argument to this function should
            only be set to that number (greater than zero) if you intend to
            train on a single frame of each example, on each minibatch.  If you
            provide this argument >0, then for each training job a different
            frame from the dumped example is selected to train on, based on
            the option --frame=n to nnet3-copy-egs.
            If you leave frames_per_eg at its default value (-1), then the
            entire sequence of frames is used for supervision.  This is suitable
            for RNN training, where it helps to amortize the cost of computing
            the activations for the frames of context needed for the recurrence.
        use_egs : True, if different examples used to train multiple
            tasks or outputs, e.g.multilingual training.  multilingual egs can
            be generated using get_egs.sh and
            subtools/kaldi/steps_multitask/nnet3/multilingual/allocate_multilingual_examples.py, those
            are the top-level scripts.
    """

    chunk_level_training = False if am_frames_per_eg > 0 else True

    deriv_time_opts = []
    if min_deriv_time is not None:
        deriv_time_opts.append("--optimization.min-deriv-time={0}".format(
                           min_deriv_time))
    if max_deriv_time_relative is not None:
        deriv_time_opts.append("--optimization.max-deriv-time-relative={0}".format(
                           max_deriv_time_relative))

    threads = []

    # the GPU timing info is only printed if we use the --verbose=1 flag; this
    # slows down the computation slightly, so don't accumulate it on every
    # iteration.  Don't do it on iteration 0 either, because we use a smaller
    # than normal minibatch size, and people may get confused thinking it's
    # slower for iteration 0 because of the verbose option.
    verbose_opt = ("--verbose=1" if iter % 20 == 0 and iter > 0 else "")

    # load the length of each xvector archives
    f_ark = open("{0}/archive_chunk_lengths".format(xvec_egs_dir))
    archive_chunk_lengths = [int(line.strip().split(' ')[1]) for line in f_ark.readlines()]
    f_ark.close()

    for job in range(1, num_jobs+1):
        # k is a zero-based index that we will derive the other indexes from.
        k = num_archives_processed + job - 1

        # work out the 1-based archive index.
        am_archive_index = (k % num_archives) + 1  
        xvec_archive_index = (k % (num_archives*am_frames_per_eg)) + 1

        # the num of xvector archives is frames_per_eg times the num of am archives


        [am_minibatch_size, xvec_minibatch_size] = minibatch_size_str.split(";")
        tmp_minibatch_size = "{am_egs_size}={am_minibatch_size}/{xvec_egs_size}={xvec_minibatch_size}".format(
                               am_egs_size=archive_chunk_lengths[xvec_archive_index - 1] - 1,
                               am_minibatch_size=am_minibatch_size,
                               xvec_egs_size=archive_chunk_lengths[xvec_archive_index - 1],
                               xvec_minibatch_size=xvec_minibatch_size)
        if not common_train_lib.validate_minibatch_size_str(tmp_minibatch_size):
            raise Exception("minibatch_size {0} has an invalid value".format(tmp_minibatch_size))

        if not chunk_level_training:
            frame = (k / num_archives + am_archive_index) % am_frames_per_eg

        cache_io_opts = (("--read-cache={dir}/cache.{iter}".format(dir=dir,
                                                                  iter=iter)
                          if iter > 0 else "") +
                         (" --write-cache={0}/cache.{1}".format(dir, iter + 1)
                          if job == 1 else ""))

        if image_augmentation_opts:
            image_augmentation_cmd = (
                'nnet3-egs-augment-image --srand={srand} {aug_opts} ark:- ark:- |'.format(
                    srand=k+srand,
                    aug_opts=image_augmentation_opts))
        else:
            image_augmentation_cmd = ''

        num_am_egs = int(common_lib.get_command_stdout("cat {0}/egs.{1}.scp | wc -l".format(am_egs_dir, am_archive_index)))
        num_xvec_egs = int(common_lib.get_command_stdout("cat {0}/egs.{1}.scp | wc -l".format(xvec_egs_dir, xvec_archive_index)))

        egs_rspecifier = (
            """ark,bg:nnet3-copy-cvector-egs {frame_opts} {cvector_opts} \
            ark:{am_egs_dir}/egs.{am_archive_index}.ark \
            ark:{xvec_egs_dir}/egs.{xvec_archive_index}.ark ark:- |\
            nnet3-shuffle-egs --buffer-size={shuffle_buffer_size} \
            --srand={srand} ark:- ark:- | {aug_cmd} \
            nnet3-merge-egs --minibatch-size={minibatch_size} ark:- ark:- |""".format(
                frame_opts=("" if chunk_level_training
                            else "--frame={0}".format(frame)),
                cvector_opts=("--am-weight={0} --xvec-weight={1} --num-am-egs={2} --num-xvec-egs={3} --am-output-name={4} --xvec-output-name={5}".format(am_weight, xvec_weight, num_am_egs, num_xvec_egs, am_output_name, xvec_output_name)),
                am_egs_dir=am_egs_dir,
                xvec_egs_dir=xvec_egs_dir,
                am_archive_index=am_archive_index,
                xvec_archive_index=xvec_archive_index,
                shuffle_buffer_size=shuffle_buffer_size,
                minibatch_size=tmp_minibatch_size,
                aug_cmd=image_augmentation_cmd,
                srand=iter+srand))

        # note: the thread waits on that process's completion.
        thread = common_lib.background_command(
            """{command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
                    nnet3-train {parallel_train_opts} {cache_io_opts} \
                     {verbose_opt} --print-interval=10 \
                    --momentum={momentum} \
                    --max-param-change={max_param_change} \
                    --backstitch-training-scale={backstitch_training_scale} \
                    --l2-regularize-factor={l2_regularize_factor} \
                    --backstitch-training-interval={backstitch_training_interval} \
                    --srand={srand} \
                    {deriv_time_opts} "{raw_model}" "{egs_rspecifier}" \
                    {dir}/{next_iter}.{job}.raw""".format(
                command=run_opts.command,
                train_queue_opt=run_opts.train_queue_opt,
                dir=dir, iter=iter,
                next_iter=iter + 1, srand=iter + srand,
                job=job,
                parallel_train_opts=run_opts.parallel_train_opts,
                cache_io_opts=cache_io_opts,
                verbose_opt=verbose_opt,
                momentum=momentum, max_param_change=max_param_change,
                l2_regularize_factor=1.0/num_jobs,
                backstitch_training_scale=backstitch_training_scale,
                backstitch_training_interval=backstitch_training_interval,
                deriv_time_opts=" ".join(deriv_time_opts),
                raw_model=raw_model_string,
                egs_rspecifier=egs_rspecifier),
                require_zero_status=True)

        threads.append(thread)

        control={"SleepTime":3,"Limit":8}
        if os.path.isfile(dir+"/control.conf"):
            for line in open(dir+"/control.conf","r"):
                control[line.split("=")[0]]=int(line.split("=")[1])

        print("Sleep %ds..."%(control["SleepTime"]))
        time.sleep(control["SleepTime"])

        while(len(common_lib.threading.enumerate())>=control["Limit"]+1):
            time.sleep(10)

    for thread in threads:
        thread.join()


def train_one_iteration(dir, iter, srand, egs_dir,
                        num_jobs, num_archives_processed, num_archives,
                        learning_rate, minibatch_size_str,
                        momentum, max_param_change, shuffle_buffer_size,
                        run_opts, image_augmentation_opts=None,
                        frames_per_eg=-1,
                        min_deriv_time=None, max_deriv_time_relative=None,
                        shrinkage_value=1.0, dropout_edit_string="",
                        get_raw_nnet_from_am=True,
                        use_egs=False,
                        backstitch_training_scale=0.0, backstitch_training_interval=1,
                        compute_per_dim_accuracy=False):
    """ Called from subtools/kaldi/steps_multitask/nnet3/train_*.py scripts for one iteration of neural
    network training

    Selected args:
        frames_per_eg: The default value -1 implies chunk_level_training, which
            is particularly applicable to RNN training. If it is > 0, then it
            implies frame-level training, which is applicable for DNN training.
            If it is > 0, then each parallel SGE job created, a different frame
            numbered 0..frames_per_eg-1 is used.
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
        get_raw_nnet_from_am: If True, then the network is read and stored as
            acoustic model i.e. along with transition model e.g. 10.mdl
            as against a raw network e.g. 10.raw when the value is False.
    """

    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    logger.info("Training neural net (pass {0})".format(iter))

    # check if different iterations use the same random seed
    if os.path.exists('{0}/srand'.format(dir)):
        try:
            saved_srand = int(open('{0}/srand'.format(dir)).readline().strip())
        except (IOError, ValueError):
            logger.error("Exception while reading the random seed "
                         "for training")
            raise
        if srand != saved_srand:
            logger.warning("The random seed provided to this iteration "
                           "(srand={0}) is different from the one saved last "
                           "time (srand={1}). Using srand={0}.".format(
                               srand, saved_srand))
    else:
        with open('{0}/srand'.format(dir), 'w') as f:
            f.write(str(srand))

    # Sets off some background jobs to compute train and
    # validation set objectives
    compute_train_cv_probabilities(
        dir=dir, iter=iter, egs_dir=egs_dir,
        run_opts=run_opts,
        get_raw_nnet_from_am=get_raw_nnet_from_am,
        use_egs=use_egs,
        compute_per_dim_accuracy=compute_per_dim_accuracy)

    if iter > 0:
        # Runs in the background
        # egs_dir takes no effect in this function 
        compute_progress(dir=dir, iter=iter, egs_dir=egs_dir,
                         run_opts=run_opts,
                         get_raw_nnet_from_am=get_raw_nnet_from_am)

    do_average = (iter > 0)


    raw_model_string = ("nnet3-copy --learning-rate={lr} --scale={s} "
                        "{dir}/{iter}.{suf} - |".format(
                            lr=learning_rate, s=shrinkage_value,
                            suf="mdl" if get_raw_nnet_from_am else "raw",
                            dir=dir, iter=iter))

    raw_model_string = raw_model_string + dropout_edit_string

    if do_average:
        cur_minibatch_size_str = minibatch_size_str
        cur_max_param_change = max_param_change
    else:
        # on iteration zero, use a smaller minibatch size (and we will later
        # choose the output of just one of the jobs): the model-averaging isn't
        # always helpful when the model is changing too fast (i.e. it can worsen
        # the objective function), and the smaller minibatch size will help to
        # keep the update stable.
        cur_minibatch_size_str = common_train_lib.halve_minibatch_size_str(minibatch_size_str)
        cur_max_param_change = float(max_param_change) / math.sqrt(2)

    shrink_info_str = ''
    if shrinkage_value != 1.0:
        shrink_info_str = ' and shrink value is {0}'.format(shrinkage_value)

    logger.info("On iteration {0}, learning rate is {1}"
                "{shrink_info}.".format(
                    iter, learning_rate,
                    shrink_info=shrink_info_str))

    train_new_models(dir=dir, iter=iter, srand=srand, num_jobs=num_jobs,
                     num_archives_processed=num_archives_processed,
                     num_archives=num_archives,
                     raw_model_string=raw_model_string, egs_dir=egs_dir,
                     momentum=momentum, max_param_change=cur_max_param_change,
                     shuffle_buffer_size=shuffle_buffer_size,
                     minibatch_size_str=cur_minibatch_size_str,
                     run_opts=run_opts,
                     frames_per_eg=frames_per_eg,
                     min_deriv_time=min_deriv_time,
                     max_deriv_time_relative=max_deriv_time_relative,
                     image_augmentation_opts=image_augmentation_opts,
                     use_egs=use_egs,
                     backstitch_training_scale=backstitch_training_scale,
                     backstitch_training_interval=backstitch_training_interval)

    [models_to_average, best_model] = common_train_lib.get_successful_models(
         num_jobs, '{0}/log/train.{1}.%.log'.format(dir, iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        common_train_lib.get_average_nnet_model(
            dir=dir, iter=iter,
            nnets_list=" ".join(nnets_list),
            run_opts=run_opts,
            get_raw_nnet_from_am=get_raw_nnet_from_am)

    else:
        # choose the best model from different jobs
        common_train_lib.get_best_nnet_model(
            dir=dir, iter=iter,
            best_model_index=best_model,
            run_opts=run_opts,
            get_raw_nnet_from_am=get_raw_nnet_from_am)

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        logger.error("Error while trying to delete the raw models")
        raise

    if get_raw_nnet_from_am:
        new_model = "{0}/{1}.mdl".format(dir, iter + 1)
    else:
        new_model = "{0}/{1}.raw".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of "
                        "iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in "
                        "iteration {1}".format(new_model, iter))
    if os.path.exists("{0}/cache.{1}".format(dir, iter)):
        os.remove("{0}/cache.{1}".format(dir, iter))


def train_cvector_one_iteration(dir, iter, srand, 
                        am_output_name, am_weight, am_egs_dir, 
                        xvec_output_name, xvec_weight, xvec_egs_dir,
                        num_jobs, num_archives_processed, num_archives,
                        learning_rate, minibatch_size_str,
                        momentum, max_param_change, shuffle_buffer_size,
                        run_opts, image_augmentation_opts=None,
                        am_frames_per_eg=-1,
                        min_deriv_time=None, max_deriv_time_relative=None,
                        shrinkage_value=1.0, dropout_edit_string="",
                        get_raw_nnet_from_am=True,
                        use_egs=False,
                        backstitch_training_scale=0.0, backstitch_training_interval=1,
                        compute_per_dim_accuracy=False):
    """ Called from subtools/kaldi/steps_multitask/nnet3/train_*.py scripts for one iteration of neural
    network training

    Selected args:
        am_frames_per_eg: The default value -1 implies chunk_level_training, which
            is particularly applicable to RNN training. If it is > 0, then it
            implies frame-level training, which is applicable for DNN training.
            If it is > 0, then each parallel SGE job created, a different frame
            numbered 0..frames_per_eg-1 is used.
        shrinkage_value: If value is 1.0, no shrinkage is done; otherwise
            parameter values are scaled by this value.
        get_raw_nnet_from_am: If True, then the network is read and stored as
            acoustic model i.e. along with transition model e.g. 10.mdl
            as against a raw network e.g. 10.raw when the value is False.
    """

    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    logger.info("Training neural net (pass {0})".format(iter))

    # check if different iterations use the same random seed
    if os.path.exists('{0}/srand'.format(dir)):
        try:
            saved_srand = int(open('{0}/srand'.format(dir)).readline().strip())
        except (IOError, ValueError):
            logger.error("Exception while reading the random seed "
                         "for training")
            raise
        if srand != saved_srand:
            logger.warning("The random seed provided to this iteration "
                           "(srand={0}) is different from the one saved last "
                           "time (srand={1}). Using srand={0}.".format(
                               srand, saved_srand))
    else:
        with open('{0}/srand'.format(dir), 'w') as f:
            f.write(str(srand))

    # Sets off some background jobs to compute train and
    # validation set objectives
    compute_cvector_train_cv_probabilities(
        dir=dir, iter=iter, am_output_name=am_output_name, am_weight=am_weight, am_egs_dir=am_egs_dir, 
        xvec_output_name=xvec_output_name, xvec_weight=xvec_weight, xvec_egs_dir=xvec_egs_dir,
        run_opts=run_opts,
        get_raw_nnet_from_am=get_raw_nnet_from_am,
        compute_per_dim_accuracy=compute_per_dim_accuracy)

    if iter > 0:
        # Runs in the background
        compute_cvector_progress(dir=dir, iter=iter, 
                         run_opts=run_opts,
                         get_raw_nnet_from_am=get_raw_nnet_from_am)

    do_average = (iter > 0)


    raw_model_string = ("nnet3-copy --learning-rate={lr} --scale={s} "
                        "{dir}/{iter}.{suf} - |".format(
                            lr=learning_rate, s=shrinkage_value,
                            suf="mdl" if get_raw_nnet_from_am else "raw",
                            dir=dir, iter=iter))

    raw_model_string = raw_model_string + dropout_edit_string

    if do_average:
        cur_minibatch_size_str = minibatch_size_str
        cur_max_param_change = max_param_change
    else:
        # on iteration zero, use a smaller minibatch size (and we will later
        # choose the output of just one of the jobs): the model-averaging isn't
        # always helpful when the model is changing too fast (i.e. it can worsen
        # the objective function), and the smaller minibatch size will help to
        # keep the update stable.
        cur_minibatch_size_str = common_train_lib.halve_cvector_minibatch_size_str(minibatch_size_str)
        cur_max_param_change = float(max_param_change) / math.sqrt(2)

    shrink_info_str = ''
    if shrinkage_value != 1.0:
        shrink_info_str = ' and shrink value is {0}'.format(shrinkage_value)

    logger.info("On iteration {0}, learning rate is {1}"
                "{shrink_info}.".format(
                    iter, learning_rate,
                    shrink_info=shrink_info_str))

    train_cvector_new_models(dir=dir, iter=iter, srand=srand, num_jobs=num_jobs,
                     num_archives_processed=num_archives_processed,
                     num_archives=num_archives,
                     raw_model_string=raw_model_string, 
                     am_output_name=am_output_name,
                     am_weight=am_weight,
                     am_egs_dir=am_egs_dir,
                     xvec_output_name=xvec_output_name,
                     xvec_weight=xvec_weight,
                     xvec_egs_dir=xvec_egs_dir,
                     momentum=momentum, max_param_change=cur_max_param_change,
                     shuffle_buffer_size=shuffle_buffer_size,
                     minibatch_size_str=cur_minibatch_size_str,
                     run_opts=run_opts,
                     am_frames_per_eg=am_frames_per_eg,
                     min_deriv_time=min_deriv_time,
                     max_deriv_time_relative=max_deriv_time_relative,
                     image_augmentation_opts=image_augmentation_opts,
                     use_egs=use_egs,
                     backstitch_training_scale=backstitch_training_scale,
                     backstitch_training_interval=backstitch_training_interval)

    [models_to_average, best_model] = common_train_lib.get_successful_models(
         num_jobs, '{0}/log/train.{1}.%.log'.format(dir, iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        common_train_lib.get_average_nnet_model(
            dir=dir, iter=iter,
            nnets_list=" ".join(nnets_list),
            run_opts=run_opts,
            get_raw_nnet_from_am=get_raw_nnet_from_am)

    else:
        # choose the best model from different jobs
        common_train_lib.get_best_nnet_model(
            dir=dir, iter=iter,
            best_model_index=best_model,
            run_opts=run_opts,
            get_raw_nnet_from_am=get_raw_nnet_from_am)

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        logger.error("Error while trying to delete the raw models")
        raise

    if get_raw_nnet_from_am:
        new_model = "{0}/{1}.mdl".format(dir, iter + 1)
    else:
        new_model = "{0}/{1}.raw".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of "
                        "iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in "
                        "iteration {1}".format(new_model, iter))
    if os.path.exists("{0}/cache.{1}".format(dir, iter)):
        os.remove("{0}/cache.{1}".format(dir, iter))


def compute_preconditioning_matrix(dir, egs_dir, num_lda_jobs, run_opts,
                                   max_lda_jobs=None, rand_prune=4.0,
                                   lda_opts=None):
    if max_lda_jobs is not None:
        if num_lda_jobs > max_lda_jobs:
            num_lda_jobs = max_lda_jobs

    # Write stats with the same format as stats for LDA.
    common_lib.execute_command(
        """{command} JOB=1:{num_lda_jobs} {dir}/log/get_lda_stats.JOB.log \
                nnet3-acc-lda-stats --rand-prune={rand_prune} \
                {dir}/init.raw "ark:{egs_dir}/egs.JOB.ark" \
                {dir}/JOB.lda_stats""".format(
                    command=run_opts.command,
                    num_lda_jobs=num_lda_jobs,
                    dir=dir,
                    egs_dir=egs_dir,
                    rand_prune=rand_prune))

    # the above command would have generated dir/{1..num_lda_jobs}.lda_stats
    lda_stat_files = map(lambda x: '{0}/{1}.lda_stats'.format(dir, x),
                         range(1, num_lda_jobs + 1))

    common_lib.execute_command(
        """{command} {dir}/log/sum_transform_stats.log \
                sum-lda-accs {dir}/lda_stats {lda_stat_files}""".format(
                    command=run_opts.command,
                    dir=dir, lda_stat_files=" ".join(lda_stat_files)))

    for file in lda_stat_files:
        try:
            os.remove(file)
        except OSError:
            logger.error("There was error while trying to remove "
                         "lda stat files.")
            raise
    # this computes a fixed affine transform computed in the way we described
    # in Appendix C.6 of http://arxiv.org/pdf/1410.7455v6.pdf; it's a scaled
    # variant of an LDA transform but without dimensionality reduction.

    common_lib.execute_command(
        """{command} {dir}/log/get_transform.log \
                nnet-get-feature-transform {lda_opts} {dir}/lda.mat \
                {dir}/lda_stats""".format(
                    command=run_opts.command, dir=dir,
                    lda_opts=lda_opts if lda_opts is not None else ""))

    common_lib.force_symlink("../lda.mat", "{0}/configs/lda.mat".format(dir))


def compute_train_cv_probabilities(dir, iter, egs_dir, run_opts,
                                   get_raw_nnet_from_am=True,
                                   use_egs=False,
                                   compute_per_dim_accuracy=False):
    if get_raw_nnet_from_am:
        model = "nnet3-am-copy --raw=true {dir}/{iter}.mdl - |".format(
                    dir=dir, iter=iter)
    else:
        model = "{dir}/{iter}.raw".format(dir=dir, iter=iter)

    scp_or_ark = "scp" if use_egs else "ark"
    egs_suffix = ".scp" if use_egs else ".egs"
    egs_rspecifier = ("{0}:{1}/valid_diagnostic{2}".format(
        scp_or_ark, egs_dir, egs_suffix))

    opts = []
    if compute_per_dim_accuracy:
        opts.append("--compute-per-dim-accuracy")

    multitask_egs_opts = common_train_lib.get_egs_opts(
                             egs_dir,
                             egs_prefix="valid_diagnostic.",
                             use_egs=use_egs)

    common_lib.background_command(
        """ {command} {dir}/log/compute_prob_valid.{iter}.log \
                nnet3-compute-prob "{model}" \
                "ark,bg:nnet3-copy-egs {multitask_egs_opts} \
                    {egs_rspecifier} ark:- | \
                    nnet3-merge-egs --minibatch-size=1:64 ark:- \
                    ark:- |" """.format(command=run_opts.command,
                                        dir=dir,
                                        iter=iter,
                                        egs_rspecifier=egs_rspecifier,
                                        opts=' '.join(opts), model=model,
                                        multitask_egs_opts=multitask_egs_opts))

    egs_rspecifier = ("{0}:{1}/train_diagnostic{2}".format(
        scp_or_ark, egs_dir, egs_suffix))

    multitask_egs_opts = common_train_lib.get_egs_opts(
                             egs_dir,
                             egs_prefix="train_diagnostic.",
                             use_egs=use_egs)

    common_lib.background_command(
        """{command} {dir}/log/compute_prob_train.{iter}.log \
                nnet3-compute-prob {opts} "{model}" \
                "ark,bg:nnet3-copy-egs {multitask_egs_opts} \
                    {egs_rspecifier} ark:- | \
                    nnet3-merge-egs --minibatch-size=1:64 ark:- \
                    ark:- |" """.format(command=run_opts.command,
                                        dir=dir,
                                        iter=iter,
                                        egs_rspecifier=egs_rspecifier,
                                        opts=' '.join(opts), model=model,
                                        multitask_egs_opts=multitask_egs_opts))


def compute_cvector_train_cv_probabilities(dir, iter, 
                                   am_output_name, am_weight, am_egs_dir, 
                                   xvec_output_name, xvec_weight, xvec_egs_dir, 
                                   run_opts,
                                   get_raw_nnet_from_am=True,
                                   compute_per_dim_accuracy=False):
    if get_raw_nnet_from_am:
        model = "nnet3-am-copy --raw=true {dir}/{iter}.mdl - |".format(
                    dir=dir, iter=iter)
    else:
        model = "{dir}/{iter}.raw".format(dir=dir, iter=iter)

    opts = []
    if compute_per_dim_accuracy:
        opts.append("--compute-per-dim-accuracy")

    num_am_egs = int(common_lib.get_command_stdout("cat {0}/valid_diagnostic.scp | wc -l".format(am_egs_dir)))
    num_xvec_egs = int(common_lib.get_command_stdout("cat {0}/valid_diagnostic.scp | wc -l".format(xvec_egs_dir)))

    common_lib.background_command(
        """ {command} {dir}/log/compute_prob_valid.{iter}.log \
                nnet3-compute-prob "{model}" \
                "ark,bg:nnet3-copy-cvector-egs --num-am-egs={num_am_egs} --am-output-name={am_output_name} --am-weight={am_weight} \
                    --num-xvec-egs={num_xvec_egs} --xvec-output-name={xvec_output_name} --xvec-weight={xvec_weight} \
                    scp:{am_egs_dir}/valid_diagnostic.scp scp:{xvec_egs_dir}/valid_diagnostic.scp ark:- | \
                    nnet3-merge-egs --minibatch-size=1:64 ark:- \
                    ark:- |" """.format(command=run_opts.command,
                                        dir=dir,
                                        iter=iter,
                                        num_am_egs=num_am_egs,
                                        am_output_name=am_output_name,
                                        am_weight=am_weight,
                                        am_egs_dir=am_egs_dir,
                                        num_xvec_egs=num_xvec_egs,
                                        xvec_output_name=xvec_output_name,
                                        xvec_weight=xvec_weight,
                                        xvec_egs_dir=xvec_egs_dir,
                                        opts=' '.join(opts), model=model))

    num_am_egs = int(common_lib.get_command_stdout("cat {0}/train_diagnostic.scp | wc -l".format(am_egs_dir)))
    num_xvec_egs = int(common_lib.get_command_stdout("cat {0}/train_diagnostic.scp | wc -l".format(xvec_egs_dir)))

    common_lib.background_command(
        """{command} {dir}/log/compute_prob_train.{iter}.log \
                nnet3-compute-prob {opts} "{model}" \
                "ark,bg:nnet3-copy-cvector-egs --num-am-egs={num_am_egs} --am-output-name={am_output_name} --am-weight={am_weight} \
                    --num-xvec-egs={num_xvec_egs} --xvec-output-name={xvec_output_name} --xvec-weight={xvec_weight} \
                    scp:{am_egs_dir}/train_diagnostic.scp scp:{xvec_egs_dir}/train_diagnostic.scp ark:- | \
                    nnet3-merge-egs --minibatch-size=1:64 ark:- \
                    ark:- |" """.format(command=run_opts.command,
                                        dir=dir,
                                        iter=iter,
                                        num_am_egs=num_am_egs,
                                        am_output_name=am_output_name,
                                        am_weight=am_weight,
                                        am_egs_dir=am_egs_dir,
                                        num_xvec_egs=num_xvec_egs,
                                        xvec_output_name=xvec_output_name,
                                        xvec_weight=xvec_weight,
                                        xvec_egs_dir=xvec_egs_dir,
                                        opts=' '.join(opts), model=model))


def compute_progress(dir, iter, egs_dir,
                     run_opts,
                     get_raw_nnet_from_am=True):
    suffix = "mdl" if get_raw_nnet_from_am else "raw"
    prev_model = '{0}/{1}.{2}'.format(dir, iter - 1, suffix)
    model = '{0}/{1}.{2}'.format(dir, iter, suffix)

    common_lib.background_command(
            """{command} {dir}/log/progress.{iter}.log \
                    nnet3-info {model} '&&' \
                    nnet3-show-progress --use-gpu=no {prev_model} {model} """
        ''.format(command=run_opts.command, dir=dir,
                  iter=iter, model=model, prev_model=prev_model))


def compute_cvector_progress(dir, iter, 
                     run_opts,
                     get_raw_nnet_from_am=True):
    suffix = "mdl" if get_raw_nnet_from_am else "raw"
    prev_model = '{0}/{1}.{2}'.format(dir, iter - 1, suffix)
    model = '{0}/{1}.{2}'.format(dir, iter, suffix)

    common_lib.background_command(
            """{command} {dir}/log/progress.{iter}.log \
                    nnet3-info {model} '&&' \
                    nnet3-show-progress --use-gpu=no {prev_model} {model} """
        ''.format(command=run_opts.command, dir=dir,
                  iter=iter, model=model, prev_model=prev_model))


# def combine_models(dir, num_iters, models_to_combine, 
#                    egs_dir,
#                    minibatch_size_str,
#                    run_opts,
#                    chunk_width=None, get_raw_nnet_from_am=True,
#                    sum_to_one_penalty=0.0,
#                    use_egs=False,
#                    compute_per_dim_accuracy=False):
def combine_models(dir, num_iters, models_to_combine, egs_dir,
                   minibatch_size_str,
                   run_opts,
                   chunk_width=None, get_raw_nnet_from_am=True,
                   max_objective_evaluations=30,
                   use_egs=False,
                   compute_per_dim_accuracy=False):
    """ Function to do model combination

    In the nnet3 setup, the logic
    for doing averaging of subsets of the models in the case where
    there are too many models to reliably esetimate interpolation
    factors (max_models_combine) is moved into the nnet3-combine.
    """
    raw_model_strings = []
    logger.info("Combining {0} models.".format(models_to_combine))

    models_to_combine.add(num_iters)

    for iter in sorted(models_to_combine):
        suffix = "mdl" if get_raw_nnet_from_am else "raw"
        model_file = '{0}/{1}.{2}'.format(dir, iter, suffix)
        if not os.path.exists(model_file):
            raise Exception('Model file {0} missing'.format(model_file))
        raw_model_strings.append(model_file)

    if get_raw_nnet_from_am:
        out_model = ("| nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl "
                     "{dir}/combined.mdl".format(dir=dir, num_iters=num_iters))
    else:
        out_model = '{dir}/final.raw'.format(dir=dir)


    # We reverse the order of the raw model strings so that the freshest one
    # goes first.  This is important for systems that include batch
    # normalization-- it means that the freshest batch-norm stats are used.
    # Since the batch-norm stats are not technically parameters, they are not
    # combined in the combination code, they are just obtained from the first
    # model.
    raw_model_strings = list(reversed(raw_model_strings))

    scp_or_ark = "scp" if use_egs else "ark"
    egs_suffix = ".scp" if use_egs else ".egs"

    egs_rspecifier = "{0}:{1}/combine{2}".format(scp_or_ark,
                                                 egs_dir, egs_suffix)

    multitask_egs_opts = common_train_lib.get_egs_opts(
                             egs_dir,
                             egs_prefix="combine.",
                             use_egs=use_egs)
    # common_lib.execute_command(
    #     """{command} {combine_queue_opt} {dir}/log/combine.log \
    #             nnet3-combine --num-iters=80 \
    #             --enforce-sum-to-one={hard_enforce} \
    #             --sum-to-one-penalty={penalty} \
    #             --enforce-positive-weights=true \
    #             --verbose=3 {raw_models} \
    #             "ark,bg:nnet3-copy-egs {multitask_egs_opts} \
    #                 {egs_rspecifier} ark:- | \
    #                   nnet3-merge-egs --minibatch-size=1:{mbsize} ark:- ark:- |" \
    #             "{out_model}"
    #     """.format(command=run_opts.command,
    #                combine_queue_opt=run_opts.combine_queue_opt,
    #                dir=dir, raw_models=" ".join(raw_model_strings),
    #                egs_rspecifier=egs_rspecifier,
    #                hard_enforce=(sum_to_one_penalty <= 0),
    #                penalty=sum_to_one_penalty,
    #                mbsize=minibatch_size_str,
    #                out_model=out_model,
    #                multitask_egs_opts=multitask_egs_opts))

    common_lib.execute_command(
        """{command} {combine_queue_opt} {dir}/log/combine.log \
                nnet3-combine \
                --max-objective-evaluations={max_objective_evaluations} \
                --verbose=3 {raw_models} \
                "ark,bg:nnet3-copy-egs {multitask_egs_opts} \
                   {egs_rspecifier} ark:- | \
                     nnet3-merge-egs --minibatch-size=1:{mbsize} ark:- ark:- |" \
                "{out_model}"
        """.format(command=run_opts.command,
                   combine_queue_opt=run_opts.combine_queue_opt,
                   dir=dir, raw_models=" ".join(raw_model_strings),
                   max_objective_evaluations=max_objective_evaluations,
                   egs_rspecifier=egs_rspecifier,
                   mbsize=minibatch_size_str,
                   out_model=out_model,
                   multitask_egs_opts=multitask_egs_opts))

    # Compute the probability of the final, combined model with
    # the same subset we used for the previous compute_probs, as the
    # different subsets will lead to different probs.
    if get_raw_nnet_from_am:
        compute_train_cv_probabilities(
            dir=dir, iter='combined', egs_dir=egs_dir,
            run_opts=run_opts, use_egs=use_egs,
            compute_per_dim_accuracy=compute_per_dim_accuracy)
    else:
        compute_train_cv_probabilities(
            dir=dir, iter='final', egs_dir=egs_dir,
            run_opts=run_opts, get_raw_nnet_from_am=False,
            use_egs=use_egs,
            compute_per_dim_accuracy=compute_per_dim_accuracy)


def combine_cvector_models(dir, num_iters, models_to_combine, 
                   am_output_name,
                   am_weight,
                   am_egs_dir,
                   xvec_output_name,
                   xvec_weight,
                   xvec_egs_dir,
                   minibatch_size_str,
                   run_opts,
                   chunk_width=None, get_raw_nnet_from_am=True,
                   sum_to_one_penalty=0.0,
                   use_egs=False,
                   compute_per_dim_accuracy=False):
    """ Function to do model combination

    In the nnet3 setup, the logic
    for doing averaging of subsets of the models in the case where
    there are too many models to reliably esetimate interpolation
    factors (max_models_combine) is moved into the nnet3-combine.
    """
    raw_model_strings = []
    logger.info("Combining {0} models.".format(models_to_combine))

    models_to_combine.add(num_iters)

    for iter in sorted(models_to_combine):
        suffix = "mdl" if get_raw_nnet_from_am else "raw"
        model_file = '{0}/{1}.{2}'.format(dir, iter, suffix)
        if not os.path.exists(model_file):
            raise Exception('Model file {0} missing'.format(model_file))
        raw_model_strings.append(model_file)

    if get_raw_nnet_from_am:
        out_model = ("| nnet3-am-copy --set-raw-nnet=- {dir}/{num_iters}.mdl "
                     "{dir}/combined.mdl".format(dir=dir, num_iters=num_iters))
    else:
        out_model = '{dir}/final.raw'.format(dir=dir)


    # We reverse the order of the raw model strings so that the freshest one
    # goes first.  This is important for systems that include batch
    # normalization-- it means that the freshest batch-norm stats are used.
    # Since the batch-norm stats are not technically parameters, they are not
    # combined in the combination code, they are just obtained from the first
    # model.
    raw_model_strings = list(reversed(raw_model_strings))

    num_am_egs = int(common_lib.get_command_stdout("cat {0}/combine.scp | wc -l".format(am_egs_dir)))
    num_xvec_egs = int(common_lib.get_command_stdout("cat {0}/combine.scp | wc -l".format(xvec_egs_dir)))

    common_lib.execute_command(
        """{command} {combine_queue_opt} {dir}/log/combine.log \
                nnet3-combine --num-iters=80 \
                --enforce-sum-to-one={hard_enforce} \
                --sum-to-one-penalty={penalty} \
                --enforce-positive-weights=true \
                --verbose=3 {raw_models} \
                "ark,bg:nnet3-copy-cvector-egs {cvector_opts} \
                    scp:{am_egs_dir}/combine.scp scp:{xvec_egs_dir}/combine.scp ark:- | \
                      nnet3-merge-egs --minibatch-size=1:{mbsize} ark:- ark:- |" \
                "{out_model}"
        """.format(command=run_opts.command,
                   cvector_opts=("--am-weight={0} --xvec-weight={1} --num-am-egs={2} --num-xvec-egs={3} --am-output-name={4} --xvec-output-name={5}".format(am_weight, xvec_weight, num_am_egs, num_xvec_egs, am_output_name, xvec_output_name)),
                   combine_queue_opt=run_opts.combine_queue_opt,
                   dir=dir, raw_models=" ".join(raw_model_strings),
                   am_egs_dir=am_egs_dir,
                   xvec_egs_dir=xvec_egs_dir,
                   hard_enforce=(sum_to_one_penalty <= 0),
                   penalty=sum_to_one_penalty,
                   mbsize=minibatch_size_str,
                   out_model=out_model))

    # Compute the probability of the final, combined model with
    # the same subset we used for the previous compute_probs, as the
    # different subsets will lead to different probs.
    if get_raw_nnet_from_am:
        raise RuntimeError("get_raw_nnet_from_am = True? seem to be incorrect")
#        compute_cvector_train_cv_probabilities(
#            dir=dir, iter='combined', egs_dir=egs_dir,
#            run_opts=run_opts, use_egs=use_egs,
#            compute_per_dim_accuracy=compute_per_dim_accuracy)
    else:
        compute_cvector_train_cv_probabilities(
            dir=dir, iter='final', 
            am_output_name=am_output_name, am_weight=am_weight, am_egs_dir=am_egs_dir,
            xvec_output_name=xvec_output_name, xvec_weight=xvec_weight, xvec_egs_dir=xvec_egs_dir,
            run_opts=run_opts, get_raw_nnet_from_am=False,
            compute_per_dim_accuracy=compute_per_dim_accuracy)


def get_realign_iters(realign_times, num_iters,
                      num_jobs_initial, num_jobs_final):
    """ Takes the realign_times string and identifies the approximate
        iterations at which realignments have to be done.

    realign_times is a space seperated string of values between 0 and 1
    """

    realign_iters = []
    for realign_time in realign_times.split():
        realign_time = float(realign_time)
        assert(realign_time > 0 and realign_time < 1)
        if num_jobs_initial == num_jobs_final:
            realign_iter = int(0.5 + num_iters * realign_time)
        else:
            realign_iter = math.sqrt((1 - realign_time)
                                     * math.pow(num_jobs_initial, 2)
                                     + realign_time * math.pow(num_jobs_final,
                                                               2))
            realign_iter = realign_iter - num_jobs_initial
            realign_iter = realign_iter / (num_jobs_final - num_jobs_initial)
            realign_iter = realign_iter * num_iters
        realign_iters.append(int(realign_iter))

    return realign_iters


def align(dir, data, lang, run_opts, iter=None, transform_dir=None,
          online_ivector_dir=None):

    alidir = '{dir}/ali{ali_suffix}'.format(
            dir=dir,
            ali_suffix="_iter_{0}".format(iter) if iter is not None else "")

    logger.info("Aligning the data{gpu}with {num_jobs} jobs.".format(
        gpu=" using gpu " if run_opts.realign_use_gpu else " ",
        num_jobs=run_opts.realign_num_jobs))
    common_lib.execute_command(
        """subtools/kaldi/steps_multitask/nnet3/align.sh --nj {num_jobs_align} \
                --cmd "{align_cmd} {align_queue_opt}" \
                --use-gpu {align_use_gpu} \
                --transform-dir "{transform_dir}" \
                --online-ivector-dir "{online_ivector_dir}" \
                --iter "{iter}" {data} {lang} {dir} {alidir}""".format(
                    dir=dir, align_use_gpu=("yes"
                                            if run_opts.realign_use_gpu
                                            else "no"),
                    align_cmd=run_opts.realign_command,
                    align_queue_opt=run_opts.realign_queue_opt,
                    num_jobs_align=run_opts.realign_num_jobs,
                    transform_dir=(transform_dir
                                   if transform_dir is not None
                                   else ""),
                    online_ivector_dir=(online_ivector_dir
                                        if online_ivector_dir is not None
                                        else ""),
                    iter=iter if iter is not None else "",
                    alidir=alidir,
                    lang=lang, data=data))
    return alidir


def realign(dir, iter, feat_dir, lang, prev_egs_dir, cur_egs_dir,
            prior_subset_size, num_archives,
            run_opts, transform_dir=None, online_ivector_dir=None):
    raise Exception("Realignment stage has not been implemented in nnet3")
    logger.info("Getting average posterior for purposes of adjusting "
                "the priors.")
    # Note: this just uses CPUs, using a smallish subset of data.
    # always use the first egs archive, which makes the script simpler;
    # we're using different random subsets of it.

    avg_post_vec_file = compute_average_posterior(
            dir=dir, iter=iter, egs_dir=prev_egs_dir,
            num_archives=num_archives, prior_subset_size=prior_subset_size,
            run_opts=run_opts)

    avg_post_vec_file = "{dir}/post.{iter}.vec".format(dir=dir, iter=iter)
    logger.info("Re-adjusting priors based on computed posteriors")
    model = '{0}/{1}.mdl'.format(dir, iter)
    adjust_am_priors(dir, model, avg_post_vec_file, model, run_opts)

    alidir = align(dir, feat_dir, lang, run_opts, iter,
                   transform_dir, online_ivector_dir)
    common_lib.execute_command(
        """subtools/kaldi/steps_multitask/nnet3/relabel_egs.sh --cmd "{command}" --iter {iter} \
                {alidir} {prev_egs_dir} {cur_egs_dir}""".format(
                    command=run_opts.command,
                    iter=iter,
                    dir=dir,
                    alidir=alidir,
                    prev_egs_dir=prev_egs_dir,
                    cur_egs_dir=cur_egs_dir))


def adjust_am_priors(dir, input_model, avg_posterior_vector, output_model,
                     run_opts):
    common_lib.execute_command(
        """{command} {dir}/log/adjust_priors.final.log \
                nnet3-am-adjust-priors "{input_model}" {avg_posterior_vector} \
                "{output_model}" """.format(
                    command=run_opts.command,
                    dir=dir, input_model=input_model,
                    avg_posterior_vector=avg_posterior_vector,
                    output_model=output_model))


def compute_average_posterior(dir, iter, egs_dir, num_archives,
                              prior_subset_size,
                              run_opts, get_raw_nnet_from_am=True):
    """ Computes the average posterior of the network
    """
    for file in glob.glob('{0}/post.{1}.*.vec'.format(dir, iter)):
        os.remove(file)

    if run_opts.num_jobs_compute_prior > num_archives:
        egs_part = 1
    else:
        egs_part = 'JOB'

    suffix = "mdl" if get_raw_nnet_from_am else "raw"
    model = "{0}/{1}.{2}".format(dir, iter, suffix)

    common_lib.execute_command(
        """{command} JOB=1:{num_jobs_compute_prior} {prior_queue_opt} \
                {dir}/log/get_post.{iter}.JOB.log \
                nnet3-copy-egs \
                ark:{egs_dir}/egs.{egs_part}.ark ark:- \| \
                nnet3-subset-egs --srand=JOB --n={prior_subset_size} \
                ark:- ark:- \| \
                nnet3-merge-egs --minibatch-size=128 ark:- ark:- \| \
                nnet3-compute-from-egs {prior_gpu_opt} --apply-exp=true \
                "{model}" ark:- ark:- \| \
                matrix-sum-rows ark:- ark:- \| vector-sum ark:- \
                {dir}/post.{iter}.JOB.vec""".format(
                    command=run_opts.command,
                    dir=dir, model=model,
                    num_jobs_compute_prior=run_opts.num_jobs_compute_prior,
                    prior_queue_opt=run_opts.prior_queue_opt,
                    iter=iter, prior_subset_size=prior_subset_size,
                    egs_dir=egs_dir, egs_part=egs_part,
                    prior_gpu_opt=run_opts.prior_gpu_opt))

    # make sure there is time for $dir/post.{iter}.*.vec to appear.
    time.sleep(5)
    avg_post_vec_file = "{dir}/post.{iter}.vec".format(dir=dir, iter=iter)
    common_lib.execute_command(
        """{command} {dir}/log/vector_sum.{iter}.log \
                vector-sum {dir}/post.{iter}.*.vec {output_file}
        """.format(command=run_opts.command,
                   dir=dir, iter=iter, output_file=avg_post_vec_file))

    for file in glob.glob('{0}/post.{1}.*.vec'.format(dir, iter)):
        os.remove(file)
    return avg_post_vec_file
    

