rsync \
    -avczP \
    --progress \
    --exclude="log/" \
    --exclude="out/" \
    --exclude="lib/StructurePrediction" \
    --exclude="tmp/" \
    --exclude="*/__pycache__/" \
    --exclude="__pycache__/" \
    --exclude="._.DS_Store" \
    --exclude=".DS_Store" \
    --include=".*" \
    --include="out/T*/data" \
    * \
    paratera-CPU:/public1/home/scfa2650/georgezhao/TxSEml_Addon

rsync \
    -avczP \
    --progress \
    --exclude="._.DS_Store" \
    --exclude=".DS_Store" \
    --exclude="model1/" \
    --exclude="model/" \
    out \
    paratera-CPU:/public1/home/scfa2650/georgezhao/TxSEml_Addon

rsync \
    -avczP \
    --progress \
    --dry-run \
    paratera-CPU:/public1/home/scfa2650/georgezhao/TxSEml_Addon/out/libfeatureselection/T{1,2,6} \
    out/libfeatureselection