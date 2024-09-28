# Bert-SS
python3 -u src/libbert/ss.py -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/sp_p_ss.pkl
python3 -u src/libbert/ss.py -f data/T2SE/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/sp_n_ss.pkl
python3 -u src/libbert/ss.py -f data/T2SE/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/nosp_p_ss.pkl
python3 -u src/libbert/ss.py -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/nosp_n_ss.pkl
# Bert-SA
python3 -u src/libbert/sa.py -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/sp_p_sa.pkl
python3 -u src/libbert/sa.py -f data/T2SE/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/sp_n_sa.pkl
python3 -u src/libbert/sa.py -f data/T2SE/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/nosp_p_sa.pkl
python3 -u src/libbert/sa.py -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/nosp_n_sa.pkl
# Bert-DISO
python3 -u src/libbert/diso.py -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/sp_p_diso.pkl
python3 -u src/libbert/diso.py -f data/T2SE/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/sp_n_diso.pkl
python3 -u src/libbert/diso.py -f data/T2SE/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/nosp_p_diso.pkl
python3 -u src/libbert/diso.py -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/bert/nosp_n_diso.pkl

# Expasy
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/spT2SE_expasy.json
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T2SE/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/sp_paired_non_t2se_expasy.json
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T2SE/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/nospT2SE_expasy.json
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/nosp_paired_non_t2se_expasy.json

# PC-PSEAac
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/spT2SE.fasta out/libfeatureselection/SP_feature_research/featuredb/spT2SE_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/sp_paired_non_t2se.fasta out/libfeatureselection/SP_feature_research/featuredb/sp_paired_non_t2se_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/nospT2SE.fasta out/libfeatureselection/SP_feature_research/featuredb/nospT2SE_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/nosp_paired_non_t2se.fasta out/libfeatureselection/SP_feature_research/featuredb/nosp_paired_non_t2se_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
# PC-PSEAac
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/spT2SE.fasta out/libfeatureselection/SP_feature_research/featuredb/spT2SE_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/sp_paired_non_t2se.fasta out/libfeatureselection/SP_feature_research/featuredb/sp_paired_non_t2se_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/nospT2SE.fasta out/libfeatureselection/SP_feature_research/featuredb/nospT2SE_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T2SE/nosp_paired_non_t2se.fasta out/libfeatureselection/SP_feature_research/featuredb/nosp_paired_non_t2se_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5

# Top_n_gram
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/spT2SE_topm.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T2SE/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/sp_paired_non_t2se_topm.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T2SE/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/nospT2SE_topm.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/nosp_paired_non_t2se_topm.json

# Simi-B45
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/ --db lib/ep3/db.t2.fasta -t t2
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T2SE/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/ --db lib/ep3/db.t2.fasta -t t2
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T2SE/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/ --db lib/ep3/db.t2.fasta -t t2
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/ --db lib/ep3/db.t2.fasta -t t2

# PSSM
python3 -u src/libexec/possum_submiter.py submiter -f data/T2SE/spT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/possum/possum_index.json -t spT2SE

python3 -u src/libexec/possum_submiter.py fasta -f data/T2SE/sp_paired_non_t2se.fasta -o tmp/sp_paired_non_t2se.fasta
python3 -u src/libexec/possum_submiter.py submiter -f tmp/sp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/possum/possum_index.json -t sp_paired_non_t2se

python3 -u src/libexec/possum_submiter.py fasta -f data/T2SE/nospT2SE.fasta -o tmp/nospT2SE.fasta
python3 -u src/libexec/possum_submiter.py submiter -f tmp/nospT2SE.fasta -o out/libfeatureselection/SP_feature_research/featuredb/possum/possum_index.json -t nospT2SE

python3 -u src/libexec/possum_submiter.py submiter -f data/T2SE/nosp_paired_non_t2se.fasta -o out/libfeatureselection/SP_feature_research/featuredb/possum/possum_index.json -t nosp_paired_non_t2se

python3 -u src/libexec/possum_submiter.py downloader -j out/libfeatureselection/SP_feature_research/featuredb/possum/possum_index.json -o out/libfeatureselection/SP_feature_research/featuredb/possum/