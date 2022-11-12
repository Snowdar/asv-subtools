
wav_scp=./wav.scp
model_path=./models/init.zip
feat_conf=feat_conf.yaml

position=near
do_vad=false
vad_energy_threshold=1
vad_energy_mean_scale=0.5
vad_frames_context=0
vad_proportion_threshold=0.6

vad_opts=
vad_opts="$vad_opts --do_vad=$do_vad"
vad_opts="$vad_opts --vad_energy_threshold $vad_energy_threshold"
vad_opts="$vad_opts --vad_energy_mean_scale $vad_energy_mean_scale"
vad_opts="$vad_opts --vad_frames_context $vad_frames_context"
vad_opts="$vad_opts --vad_proportion_threshold $vad_proportion_threshold"



./../build/extractor_main --wav_scp $wav_scp \
    --feat_conf $feature_conf \
    --model_path $model_path \
    $vad_opts \
    --position $position &>test.1.log