#!/bin/bash
#hyperparameter_csvfile="./hyperparameter/modified-csv_two-gpu.csv"
hyperparameter_csvfile="./hyperparameter/modified-csv_single-gpu.csv"

train_log_file="./logs/train.log"
test_log_file="./logs/test.log"
roc_log_file="./logs/roc.log"


# with singe GPU or multu-GPU ?  
gpu_ids="0"    # single
#gpu_ids="0,1" # multi


rm -f ${train_log_file}
rm -f ${test_log_file}
rm -f ${roc_log_file}

total=$(tail -n +2 "${hyperparameter_csvfile}" | wc -l)

i=1
for row in `tail -n +2 ${hyperparameter_csvfile}`; do
  model=$(echo "${row}" | cut -d "," -f1)
  optimizer=$(echo "${row}" | cut -d "," -f2)
  batch_size=$(echo "${row}" | cut -d "," -f3)
  epochs=$(echo "${row}" | cut -d "," -f4)
  image_size=$(echo "${row}" | cut -d "," -f5)


  echo "${i}/${total}: Training starts..."
  echo "model: ${model}"
  echo "optimizer: ${optimizer}"
  echo "batch_size: ${batch_size}"
  echo "epochs: ${epochs}"
  echo "image_size ${image_size}x${image_size}"


  #echo "CUDA_VISIBLE_DEVICES=0 python train.py --model ${model} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --image_size ${image_size}"
  #CUDA_VISIBLE_DEVICES=0 python train.py --model ${model} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --image_size ${image_size} |& tee -a ${train_log_file}
  
  echo "python train.py --model ${model} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --image_size ${image_size} --gpu_ids ${gpu_ids}"
  python train.py --model ${model} --optimizer ${optimizer} --epochs ${epochs} --batch_size ${batch_size} --image_size ${image_size} --gpu_ids ${gpu_ids} |& tee -a ${train_log_file}



  echo "${i}/${total}: Test starts..."
  #echo "python test.py --model ${model} --image_size ${image_size}"
  #CUDA_VISIBLE_DEVICES=0 python test.py --model ${model} --image_size ${image_size} |& tee -a ${test_log_file}

  echo "python test.py --model ${model} --image_size ${image_size} --gpu_ides ${gpu_ids}"
  python test.py --model ${model} --image_size ${image_size} --gpu_ids ${gpu_ids} |& tee -a ${test_log_file}




  echo "${i}/${total}: Plot ROC..."
  cd ../Metrics
  python ROC.py |& tee -a ../Main/${roc_log_file}
  cd ../Main
  printf "\r Index: %d/%d\n\n" ${i} ${total} # Show progress
  i=$(($i + 1))
done
