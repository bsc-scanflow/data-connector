if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/Informer" ]; then
    mkdir ./logs/Informer
fi
seq_len=10
model_name=informer

root_path_name=./load-data
data_path_name=df.csv
data_name=custom

random_seed=42
pred_len=3
python -u ./informer_vanilla/main_informer.py \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model $model_name \
  --data $data_name \
  --features MS \
  --target PredictionTimeTS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --inverse True \
  --categorical_cols node \
  --c_out 1 \
  --label_len 6 \
  --enc_in 11 \
  --dec_in 11 \
  --e_layers 2 \
  --d_model 512 \
  --dropout 0.05\
  --train_epochs 1\
  --patience 3\
  --use_gpu False \
  --itr 1 --batch_size 32 --learning_rate 0.0001 | tee logs/Informer/$model_name'_'$seq_len'_pl'$pred_len'.log'
