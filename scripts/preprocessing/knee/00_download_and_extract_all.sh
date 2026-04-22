#!/usr/bin/env bash
set -euo pipefail
# Optional: cd to your working directory where you want the files
# cd /storage/ice-shared/ae8803che/lmkh3
echo "=== fastMRI knee MULTICOIL: download + extract ==="

echo "=== Downloading knee_multicoil_train batches (0-4) ==="

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=sfGuOD%2BdhXGe16EmULgBxglSxKM%3D&Expires=1770099767" --output knee_multicoil_train_batch_0.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=yvfaruNxnfhvrNecRhqgYAyw%2FiE%3D&Expires=1770099767" --output knee_multicoil_train_batch_1.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=F2of7YNZPLKOXsudAq5NJSaRqH4%3D&Expires=1770099767" --output knee_multicoil_train_batch_2.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_3.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=X3XWTFbtBNZTLSkht0kL89%2FRtlQ%3D&Expires=1770099767" --output knee_multicoil_train_batch_3.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_4.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Jkn6qVZrTpR4MqECaQjvq55e4RU%3D&Expires=1770099767" --output knee_multicoil_train_batch_4.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=49u3FG0fcK3W4VVn9kiDhYVsNKM%3D&Expires=1770099767" --output knee_multicoil_val.tar.xz
curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=nFGhcZaGNvFnwxuz2sLLzKt6OPw%3D&Expires=1770099767" --output knee_multicoil_test.tar.xz

echo "=== Extracting knee_multicoil_train batches ==="
for i in {0..4}; do
  mkdir -p "knee_multicoil_train_batch_${i}"
  tar -xvf "knee_multicoil_train_batch_${i}.tar.xz" \
      -C "knee_multicoil_train_batch_${i}"
done

echo "=== Extracting knee_multicoil_val ==="
mkdir -p knee_multicoil_val
tar -xvf knee_multicoil_val.tar.xz -C knee_multicoil_val

echo "=== Extracting knee_multicoil_test ==="
mkdir -p knee_multicoil_test
tar -xvf knee_multicoil_test.tar.xz -C knee_multicoil_test


echo "=== All knee downloads complete ==="

echo "=== Verifying dataset contents by counting volumes and slices ==="
    echo
    done
    
    echo "TOTALS for split=${split}:"
    echo "  RAW: volumes=${total_raw_vols}  slices(pngs)=${total_raw_slices}"
    echo "  GT : volumes=${total_gt_vols}   slices(pngs)=${total_gt_slices}"
    echo
    done