from tenpy.models.aklt import AKLTChain
num_site=16
#aklt基态

aklt=AKLTChain({"J":1,"L":num_site})
#运行psi_AKLT函数报错
psi=aklt.psi_AKLT()