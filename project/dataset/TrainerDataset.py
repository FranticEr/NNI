from .SelfDataset import TableControlFullLoadDataset

class TableControlTrainerDataset(TableControlFullLoadDataset):
    def __getitem__(self, index):
        # {"data":data,"time":time,"index":idx,"KSS":KSS,"LEVEL":LEVEL,"SPEED":SPEED} 
        dataset=super().__getitem__(index)
        return dataset['data'],dataset['LEVEL']