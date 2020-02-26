GPU_ID=$1
DATASET=$2
SPLITBY=$3
MODE=$4

ID="mrcn_cmr_with_st"

case ${DATASET} in
    refcoco)
        for SPLIT in val testA testB
        do
            CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_ref.py \
                --dataset ${DATASET} \
                --splitBy ${SPLITBY} \
                --split ${SPLIT} \
                --id ${ID} \
                --mode ${MODE}
        done
    ;;
    # refcoco+)
    #     for SPLIT in val testA testB
    #     do
    #         CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
    #             --dataset ${DATASET} \
    #             --splitBy ${SPLITBY} \
    #             --split ${SPLIT} \
    #             --id ${ID}
    #     done
    # ;;
    # refcocog)
    #     for SPLIT in val test
    #     do
    #         CUDA_VISIBLE_DEVICES=${GPU_ID} python ./tools/eval_dets.py \
    #             --dataset ${DATASET} \
    #             --splitBy ${SPLITBY} \
    #             --split ${SPLIT} \
    #             --id ${ID}
    #     done
    # ;;
esac