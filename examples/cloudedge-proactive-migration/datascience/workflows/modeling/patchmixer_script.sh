if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=10
model_name=PatchMixer

root_path_name=./data/
data_path_name=df.csv
model_id_name=PatchMixer
data_name=custom

random_seed=42
pred_len=3
python -u ./modeling/PatchMixer/run_longExp.py \
  --random_seed $random_seed \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name \
  --model $model_name \
  --data $data_name \
  --features MS \
  --target PredictionTimeTS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 9 \
  --e_layers 1 \
  --d_model 256 \
  --dropout 0.2\
  --head_dropout 0\
  --patch_len 16\
  --stride 8 \
  --des 'Exp' \
  --train_epochs 15\
  --patience 5\
  --loss_flag 2\
  --use_gpu False \
  --itr 1 --batch_size 256 --learning_rate 0.001 | tee logs/LongForecasting/$model_name'_'$model_id_name'_sl'$seq_len'_pl'$pred_len'_random_seed'$random_seed.log
