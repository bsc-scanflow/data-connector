if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

root_path_name=./data/
data_path_name=df.csv

# PatchMixer parameters
enc_in=10
checkpoints=./checkpoints/
loss_flag=2
learning_rate=0.001
d_model=256
model_id_name=PatchMixer
data_name=custom
features=MS
seq_len=10
pred_len=3
patch_len=16
stride=8
random_seed=42

# Construct parameters as a JSON string
echo $parameters
python -u ./modeling/PatchMixer/run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_id_name \
  --data $data_name \
  --features $features \
  --target PredictionTimeTS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in $enc_in \
  --e_layers 1 \
  --d_model $d_model \
  --dropout 0.2\
  --head_dropout 0\
  --patch_len $patch_len\
  --stride $stride \
  --des 'Exp' \
  --train_epochs 15\
  --patience 5\
  --loss_flag $loss_flag\
  --use_gpu False \
  --itr 1 --batch_size 256 --learning_rate $learning_rate | tee logs/LongForecasting/$model_id_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log \



# Compose model_path
model_path="loss_flag_${loss_flag}_lr${learning_rate}_dm${d_model}_${model_id_name}_${data_name}_${features}_sl${seq_len}_pl${pred_len}_p${patch_len}s${stride}_random${random_seed}_0"

python -u ./modeling/mlflow_loader.py \
  --experiment_name $model_id_name \
  --checkpoints $checkpoints \
  --model_name $model_path \
  --parameters "$parameters" \