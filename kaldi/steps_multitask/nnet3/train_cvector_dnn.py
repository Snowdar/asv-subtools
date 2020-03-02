#!/usr/bin/env python

# Copyright 2016    Vijayaditya Peddinti.
#           2016    Vimal Manohar
#           2017 Johns Hopkins University (author: Daniel Povey)
#           2018 Yi Liu
# Apache 2.0.

""" This script is based on subtools/kaldi/steps_multitask/nnet3/tdnn/train.sh
"""

from __future__ import print_function
import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback

sys.path.insert(0, 'subtools/kaldi/steps_multitask')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib
import libs.nnet3.train.frame_level_objf as train_lib
import libs.nnet3.report.log_parse as nnet3_log_parse


logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting DNN trainer (train_dnn.py)')


def get_args():
    """ Get args from stdin.

    We add compulsory arguments as named arguments for readability

    The common options are defined in the object
    libs.nnet3.train.common.CommonParser.parser.
    See subtools/kaldi/steps_multitask/libs/nnet3/train/common.py
    """
    parser = argparse.ArgumentParser(
        description="""Trains a feed forward DNN acoustic model using the
        cross-entropy objective.  DNNs include simple DNNs, TDNNs and CNNs.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve',
        parents=[common_train_lib.CommonParser(include_chunk_context=False).parser])

    # Parameters for the optimization
    parser.add_argument("--trainer.optimization.minibatch-size",
                        type=str, dest='minibatch_size', default='512',
                        help="""Size of the minibatch used in SGD training
                        (argument to nnet3-merge-egs); may be a more general
                        rule as accepted by the --minibatch-size option of
                        nnet3-merge-egs; run that program without args to see
                        the format.""")
    parser.add_argument("--trainer.num-jobs-compute-prior", type=int,
                        dest='num_jobs_compute_prior', default=10,
                        help="The prior computation jobs are single "
                        "threaded and run on the CPU")

    # General options
    parser.add_argument("--am-output-name", type=str, required=True,
                        help="The name of am output-node")
    parser.add_argument("--xvec-output-name", type=str, required=True,
                        help="The name of xvec output-node")
    parser.add_argument("--am-weight", type=float, default=1.0,
                        help="The am weight")
    parser.add_argument("--xvec-weight", type=float, default=1.0,
                        help="The xvec weight")
    parser.add_argument("--am-egs-dir", type=str, required=True,
                        help="Directory with am egs for training")
    parser.add_argument("--xvec-egs-dir", type=str, required=True,
                        help="Directory with xvector egs for training")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to store the models and "
                        "all other files.")

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)

    args = parser.parse_args()

    [args, run_opts] = process_args(args)

    return [args, run_opts]


def process_args(args):
    """ Process the options got from get_args()
    """

    if (not os.path.exists(args.dir)
            or not os.path.exists(args.dir+"/configs")):
        raise Exception("This scripts expects {0} to exist and have a configs "
                        "directory which is the output of "
                        "make_configs.py script")

    # set the options corresponding to args.use_gpu
    run_opts = common_train_lib.RunOpts()
    if args.use_gpu:
        if not common_lib.check_if_cuda_compiled():
            logger.warning(
                """You are running with one thread but you have not compiled
                   for CUDA.  You may be running a setup optimized for GPUs.
                   If you have GPUs and have nvcc installed, go to src/ and do
                   ./configure; make""")

        run_opts.train_queue_opt = "--gpu 1"
        run_opts.parallel_train_opts = ""
        run_opts.combine_queue_opt = "--gpu 1"
        run_opts.prior_gpu_opt = "--use-gpu=yes"
        run_opts.prior_queue_opt = "--gpu 1"
    else:
        logger.warning("Without using a GPU this will be very slow. "
                       "nnet3 does not yet support multiple threads.")

        run_opts.train_queue_opt = ""
        run_opts.parallel_train_opts = "--use-gpu=no"
        run_opts.combine_queue_opt = ""
        run_opts.prior_gpu_opt = "--use-gpu=no"
        run_opts.prior_queue_opt = ""

    run_opts.command = args.command
    run_opts.egs_command = (args.egs_command
                            if args.egs_command is not None else
                            args.command)
    run_opts.num_jobs_compute_prior = args.num_jobs_compute_prior

    return [args, run_opts]


def train(args, run_opts):
    """ The main function for training.

    Args:
        args: a Namespace object with the required parameters
            obtained from the function process_args()
        run_opts: RunOpts object obtained from the process_args()
    """

    arg_string = pprint.pformat(vars(args))
    logger.info("Arguments for the experiment\n{0}".format(arg_string))

    # Set some variables.
    config_dir = '{0}/configs'.format(args.dir)
    am_var_file = '{0}/vars_am'.format(config_dir)
    xvec_var_file = '{0}/vars_xvec'.format(config_dir)
    am_variables = common_train_lib.parse_generic_config_vars_file(am_var_file)
    xvec_variables = common_train_lib.parse_generic_config_vars_file(xvec_var_file)

    # Set some variables.
    try:
        am_model_left_context = am_variables['model_left_context']
        am_model_right_context = am_variables['model_right_context']
        xvec_model_left_context = xvec_variables['model_left_context']
        xvec_model_right_context = xvec_variables['model_right_context']
    except KeyError as e:
        raise Exception("KeyError {0}: Variables need to be defined in "
                        "{1}".format(str(e), '{0}/configs'.format(args.dir)))

    am_left_context = am_model_left_context
    am_right_context = am_model_right_context
    xvec_left_context = xvec_model_left_context
    xvec_right_context = xvec_model_right_context

    # Initialize as "raw" nnet, prior to training the LDA-like preconditioning
    # matrix.  This first config just does any initial splicing that we do;
    # we do this as it's a convenient way to get the stats for the 'lda-like'
    # transform.
    if (args.stage <= -5) and os.path.exists(args.dir+"/configs/init.config"):
        logger.info("Initializing a basic network for estimating "
                    "preconditioning matrix")
        common_lib.execute_command(
            """{command} {dir}/log/nnet_init.log \
                    nnet3-init --srand=-2 {dir}/configs/init.config \
                    {dir}/init.raw""".format(command=run_opts.command,
                                             dir=args.dir))

    am_egs_dir = args.am_egs_dir
    xvec_egs_dir = args.xvec_egs_dir
    am_output_name = args.am_output_name
    xvec_output_name = args.xvec_output_name
    am_weight = args.am_weight
    xvec_weight = args.xvec_weight

    feat_dim = int(common_lib.get_command_stdout("cat {0}/info/feat_dim".format(am_egs_dir)))
    num_archives = int(common_lib.get_command_stdout("cat {0}/info/num_archives".format(am_egs_dir)))

    tmp_feat_dim = int(common_lib.get_command_stdout("cat {0}/info/feat_dim".format(xvec_egs_dir)))
    tmp_num_archives = int(common_lib.get_command_stdout("cat {0}/info/num_archives".format(xvec_egs_dir)))

    # frames_per_eg is no longer a parameter but load from am_egs/info/frames_per_eg
    am_frames_per_eg = int(common_lib.get_command_stdout("cat {0}/info/frames_per_eg".format(am_egs_dir)))

    if feat_dim != tmp_feat_dim or num_archives*am_frames_per_eg != tmp_num_archives:
        raise Exception('The am egs and xvec egs do not match')

    if args.num_jobs_final > num_archives:
        raise Exception('num_jobs_final cannot exceed the number of archives '
                        'in the egs directory')

    # # No need to copy files for decoding
    # common_train_lib.copy_egs_properties_to_exp_dir(am_egs_dir, args.dir)

    if args.stage <= -3 and os.path.exists(args.dir+"/configs/init.config"):
        logger.info('Computing the preconditioning matrix for input features')

        train_lib.common.compute_preconditioning_matrix(
            args.dir, egs_dir, num_archives, run_opts,
            max_lda_jobs=args.max_lda_jobs,
            rand_prune=args.rand_prune)

    if args.stage <= -1:
        logger.info("Preparing the initial network.")
        common_train_lib.prepare_initial_network(args.dir, run_opts)

    # set num_iters so that as close as possible, we process the data
    # $num_epochs times, i.e. $num_iters*$avg_num_jobs) ==
    # $num_epochs*$num_archives, where
    # avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
    num_archives_expanded = num_archives * am_frames_per_eg
    num_archives_to_process = int(args.num_epochs * num_archives_expanded)
    num_archives_processed = 0
    num_iters = ((num_archives_to_process * 2)
                 / (args.num_jobs_initial + args.num_jobs_final))

    # If do_final_combination is True, compute the set of models_to_combine.
    # Otherwise, models_to_combine will be none.
    if args.do_final_combination:
        models_to_combine = common_train_lib.get_model_combine_iters(
            num_iters, args.num_epochs,
            num_archives_expanded, args.max_models_combine,
            args.num_jobs_final)
    else:
        models_to_combine = None

    logger.info("Training will run for {0} epochs = "
                "{1} iterations".format(args.num_epochs, num_iters))

    for iter in range(num_iters):
        if (args.exit_stage is not None) and (iter == args.exit_stage):
            logger.info("Exiting early due to --exit-stage {0}".format(iter))
            return
        current_num_jobs = int(0.5 + args.num_jobs_initial
                               + (args.num_jobs_final - args.num_jobs_initial)
                               * float(iter) / num_iters)

        if args.stage <= iter:
            lrate = common_train_lib.get_learning_rate(iter, current_num_jobs,
                                                       num_iters,
                                                       num_archives_processed,
                                                       num_archives_to_process,
                                                       args.initial_effective_lrate,
                                                       args.final_effective_lrate)
            shrinkage_value = 1.0 - (args.proportional_shrink * lrate)
            if shrinkage_value <= 0.5:
                raise Exception("proportional-shrink={0} is too large, it gives "
                                "shrink-value={1}".format(args.proportional_shrink,
                                                          shrinkage_value))

            percent = num_archives_processed * 100.0 / num_archives_to_process
            epoch = (num_archives_processed * args.num_epochs
                     / num_archives_to_process)
            shrink_info_str = ''
            if shrinkage_value != 1.0:
                shrink_info_str = 'shrink: {0:0.5f}'.format(shrinkage_value)
            logger.info("Iter: {0}/{1}    "
                        "Epoch: {2:0.2f}/{3:0.1f} ({4:0.1f}% complete)    "
                        "lr: {5:0.6f}    {6}".format(iter, num_iters - 1,
                                                     epoch, args.num_epochs,
                                                     percent,
                                                     lrate, shrink_info_str))
            train_lib.common.train_cvector_one_iteration(
                dir=args.dir,
                iter=iter,
                srand=args.srand,
                am_output_name=am_output_name,
                am_weight=am_weight,
                am_egs_dir=am_egs_dir,
                xvec_output_name=xvec_output_name,
                xvec_weight=xvec_weight,
                xvec_egs_dir=xvec_egs_dir,
                num_jobs=current_num_jobs,
                num_archives_processed=num_archives_processed,
                num_archives=num_archives,
                learning_rate=lrate,
                minibatch_size_str=args.minibatch_size,
                momentum=args.momentum,
                max_param_change=args.max_param_change,
                shuffle_buffer_size=args.shuffle_buffer_size,
                run_opts=run_opts,
                am_frames_per_eg=am_frames_per_eg,
                dropout_edit_string=common_train_lib.get_dropout_edit_string(
                    args.dropout_schedule,
                    float(num_archives_processed) / num_archives_to_process,
                    iter),
                shrinkage_value=shrinkage_value,
                get_raw_nnet_from_am=False,
                backstitch_training_scale=args.backstitch_training_scale,
                backstitch_training_interval=args.backstitch_training_interval)

            if args.cleanup:
                # do a clean up everythin but the last 2 models, under certain
                # conditions
                common_train_lib.remove_model(
                    args.dir, iter-2, num_iters, models_to_combine,
                    args.preserve_model_interval,
                    get_raw_nnet_from_am=False)

            if args.email is not None:
                reporting_iter_interval = num_iters * args.reporting_interval
                if iter % reporting_iter_interval == 0:
                    # lets do some reporting
                    [report, times, data] = (
                        nnet3_log_parse.generate_acc_logprob_report(args.dir))
                    message = report
                    subject = ("Update : Expt {dir} : "
                               "Iter {iter}".format(dir=args.dir, iter=iter))
                    common_lib.send_mail(message, subject, args.email)

        num_archives_processed = num_archives_processed + current_num_jobs

    # when we do final combination, just use the xvector egs
    if args.stage <= num_iters:
        if args.do_final_combination:
            logger.info("Doing final combination to produce final.mdl")

            train_lib.common.combine_models(
                dir=args.dir, num_iters=num_iters,
                models_to_combine=models_to_combine,
                egs_dir=xvec_egs_dir,
                minibatch_size_str="64", run_opts=run_opts,
                get_raw_nnet_from_am=False,
                max_objective_evaluations=args.max_objective_evaluations,
                use_egs=True)
                # sum_to_one_penalty=args.combine_sum_to_one_penalty,
        else:
            common_lib.force_symlink("{0}.raw".format(num_iters),
                                     "{0}/final.raw".format(args.dir))
    
    if args.cleanup:
        logger.info("Cleaning up the experiment directory "
                    "{0}".format(args.dir))
        remove_egs = False

        common_train_lib.clean_nnet_dir(
            nnet_dir=args.dir, num_iters=num_iters, egs_dir=am_egs_dir,
            preserve_model_interval=args.preserve_model_interval,
            remove_egs=remove_egs,
            get_raw_nnet_from_am=False)

    # TODO: we may trace other output nodes expect for "output"
    # do some reporting
    outputs_list = common_train_lib.get_outputs_list("{0}/final.raw".format(
        args.dir), get_raw_nnet_from_am=False)
    if 'output' in outputs_list:
        [report, times, data] = nnet3_log_parse.generate_acc_logprob_report(args.dir)
        if args.email is not None:
            common_lib.send_mail(report, "Update : Expt {0} : "
                                         "complete".format(args.dir),
                                 args.email)
            with open("{dir}/accuracy.{output_name}.report".format(dir=args.dir,
                                                                   output_name="output"),
                      "w") as f:
                f.write(report)

    common_lib.execute_command("subtools/kaldi/steps/info/nnet3_dir_info.pl "
                               "{0}".format(args.dir))


def main():
    [args, run_opts] = get_args()
    try:
        train(args, run_opts)
        common_lib.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if args.email is not None:
            message = ("Training session for experiment {dir} "
                       "died due to an error.".format(dir=args.dir))
            common_lib.send_mail(message, message, args.email)
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
