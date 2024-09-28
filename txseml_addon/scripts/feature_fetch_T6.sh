# Bert-SS
python3 -u src/libbert/ss.py -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ab_p_ss.pkl
python3 -u src/libbert/ss.py -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ab_n_ss.pkl
python3 -u src/libbert/ss.py -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ae_p_ss.pkl
python3 -u src/libbert/ss.py -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ae_n_ss.pkl
# Bert-SA
python3 -u src/libbert/sa.py -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ab_p_sa.pkl
python3 -u src/libbert/sa.py -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ab_n_sa.pkl
python3 -u src/libbert/sa.py -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ae_p_sa.pkl
python3 -u src/libbert/sa.py -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ae_n_sa.pkl
# Bert-DISO
python3 -u src/libbert/diso.py -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ab_p_diso.pkl
python3 -u src/libbert/diso.py -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ab_n_diso.pkl
python3 -u src/libbert/diso.py -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ae_p_diso.pkl
python3 -u src/libbert/diso.py -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/bert/ae_n_diso.pkl

# Expasy
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/ab_p_expasy.json
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/ab_n_expasy.json
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/ae_p_expasy.json
python3 -u src/libexec/expasy_submiter.py -l 100 -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/ae_n_expasy.json

# PC-PSEAac
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-bacterial-effector_p.fasta out/libfeatureselection/A_feature_research/featuredb/ab_p_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-bacterial-effector_n.fasta out/libfeatureselection/A_feature_research/featuredb/ab_n_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-eukaryotic-effector_p.fasta out/libfeatureselection/A_feature_research/featuredb/ae_p_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-eukaryotic-effector_n.fasta out/libfeatureselection/A_feature_research/featuredb/ae_n_PCPseAAC.csv Protein PC-PseAAC -f csv -lamada 1 -w 0.5
# SC-PSEAac
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-bacterial-effector_p.fasta out/libfeatureselection/A_feature_research/featuredb/ab_p_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-bacterial-effector_n.fasta out/libfeatureselection/A_feature_research/featuredb/ab_n_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-eukaryotic-effector_p.fasta out/libfeatureselection/A_feature_research/featuredb/ae_p_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5
python3 -u /mnt/md0/Tools/PseinOne/src data/T6SE/anti-eukaryotic-effector_n.fasta out/libfeatureselection/A_feature_research/featuredb/ae_n_SCPseAAC.csv Protein SC-PseAAC -f csv -lamada 1 -w 0.5

# Top_n_gram
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/ab_p_topm.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/ab_n_topm.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/ae_p_topm.json
python3 -u src/libexec/Top_n_gram_data_submiter.py -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/ae_n_topm.json

# Simi-B45
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/ --db lib/ep3/db.t6.fasta -t t6
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/ --db lib/ep3/db.t6.fasta -t t6
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/ --db lib/ep3/db.t6.fasta -t t6
python3 -u src/libexec/pairwisealigner_controler.py -p python3 -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/ --db lib/ep3/db.t6.fasta -t t6

# PSSM
python3 -u src/libexec/possum_submiter.py submiter -f data/T6SE/anti-bacterial-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/possum/possum_index.json -t ab_p
python3 -u src/libexec/possum_submiter.py submiter -f data/T6SE/anti-bacterial-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/possum/possum_index.json -t ab_n
python3 -u src/libexec/possum_submiter.py submiter -f data/T6SE/anti-eukaryotic-effector_p.fasta -o out/libfeatureselection/A_feature_research/featuredb/possum/possum_index.json -t ae_p
python3 -u src/libexec/possum_submiter.py submiter -f data/T6SE/anti-eukaryotic-effector_n.fasta -o out/libfeatureselection/A_feature_research/featuredb/possum/possum_index.json -t ae_n

python3 -u src/libexec/possum_submiter.py downloader -j out/libfeatureselection/A_feature_research/featuredb/possum/possum_index.json -o out/libfeatureselection/A_feature_research/featuredb/possum/