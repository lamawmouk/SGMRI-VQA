#!/usr/bin/env bash
set -euo pipefail

# Optional: cd to your working directory where you want the files
# cd /storage/ice-shared/ae8803che/lmkh3
echo "=== Downloading brain_multicoil_train batches (0-9) ==="
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=NN5wn%2BtU9iO%2BNlzb1xnktdt%2FCeM%3D&Expires=1770099767" --output brain_multicoil_train_batch_0.tar.xz 
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=wM3Pa1y%2Fbi%2FJ%2FLI7YfLUG9nHyj8%3D&Expires=1770099767" --output brain_multicoil_train_batch_1.tar.xz 
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Mdl1o9VRrs8givK6sUCrLm7ySfI%3D&Expires=1770099767" --output brain_multicoil_train_batch_2.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_3.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=mWvs7fhwagvrj2eK11UUh%2FdTb0E%3D&Expires=1770099767" --output brain_multicoil_train_batch_3.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_4.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=aYjSpiBQcY%2BA3KtFFNL6hpC%2FLYQ%3D&Expires=1770099767" --output brain_multicoil_train_batch_4.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_5.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=X3M4khpuZlZCkFG1mJUd7GtLQ8k%3D&Expires=1770099767" --output brain_multicoil_train_batch_5.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_6.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=nDIMJxsEXD53zrHDzP5U5VsRYGI%3D&Expires=1770099767" --output brain_multicoil_train_batch_6.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_7.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=1iWN6sxDMDHXsBnItydBUvc01Jw%3D&Expires=1770099767" --output brain_multicoil_train_batch_7.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_8.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Vf%2BK2aW8LexJUkXFEbcqzZ%2FlXDA%3D&Expires=1770099767" --output brain_multicoil_train_batch_8.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_9.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=o21uiijG1%2FIYHL%2Bz18szlUbHVww%3D&Expires=1770099767" --output brain_multicoil_train_batch_9.tar.xz

echo "=== Downloading brain_multicoil_val batches (0, 1, 2) ==="

curl -C - \
  "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=P10B4ZoqfYfKB5WzrfIVX2otlY4%3D&Expires=1770099767" \
  --output brain_multicoil_val_batch_0.tar.xz

curl -C - \
  "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=7LHC8Z%2FoCHQPs%2FG6A5UNKJ5oUeI%3D&Expires=1770099767" \
  --output brain_multicoil_val_batch_1.tar.xz

curl -C - \
  "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=CDoQix1MNOGZ1oQZOJQ3agUt8lk%3D&Expires=1770099767" \
  --output brain_multicoil_val_batch_2.tar.xz



echo "=== Downloading brain_multicoil_test_full batches (0, 1, 2) ==="

curl -C - \
  "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_full_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=OythASDttUpB0nQEiUATOR7eUJQ%3D&Expires=1770099767" \
  --output brain_multicoil_test_full_batch_0.tar.xz

curl -C - \
  "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_full_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=0Vp0Nk5h%2FQYnz43lu19E%2Ff7hJw8%3D&Expires=1770099767" \
  --output brain_multicoil_test_full_batch_1.tar.xz

curl -C - \
  "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_full_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=VfscBgdJNcR%2FJWBhIfgBCKtx8sI%3D&Expires=1770099767" \
  --output brain_multicoil_test_full_batch_2.tar.xz



echo "=== Extracting VAL batches 0, 1, 2 ==="

for i in 0 1 2
do
    echo "Processing brain_multicoil_val_batch_${i} ..."
    mkdir -p "brain_multicoil_val_batch_${i}"
    tar -xvf "brain_multicoil_val_batch_${i}.tar.xz" \
        -C "brain_multicoil_val_batch_${i}"
    echo "Done val batch ${i}."
    echo
done


echo "=== Extracting TEST batches 0, 1, 2 ==="

for i in 0 1 2
do
    echo "Processing brain_multicoil_test_batch_${i} ..."
    mkdir -p "brain_multicoil_test_batch_${i}"
    tar -xvf "brain_multicoil_test_batch_${i}.tar.xz" \
        -C "brain_multicoil_test_batch_${i}"
    echo "Done test_batch ${i}."
    echo
done

echo "All done."
