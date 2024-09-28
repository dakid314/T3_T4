##将阴性数据处理名称，以及去除重复的数据
new = r'/mnt/md0/Public/T3_T4/new_Vibrio_proteolyticus_NBRC_13287.fasta'
new_file = open(new,'w')
dict ={}
list = []
with open(r'/mnt/md0/Public/T3_T4/Vibrio_proteolyticus_NBRC_13287.fasta','r') as file:
    for line in file:
        if line.startswith(">"):
            name = line.split('>')[1]
            dict[name] = ''
        else:
            dict[name] += line
cou = 0
for k,v in dict.items():
    if "protein_id" in k :
        a = k.split("protein_id=")[1]
        t = a.split(']')[0]
        list.append(t)
        new_file.write('>'+t+'\n'+v)
    else:
        cou+=1
    
        
print(cou)
list.sort()
new_l = []
for i in range(len(list)):
    if i < len(list)-1:
        if list[i] == list[i+1]:
            new_l.append(list[i])

new_file.close()




