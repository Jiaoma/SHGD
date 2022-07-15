from panda_dataset import PANDADataset

def return_dataset(cfg):
    if cfg.dataset_name=='PANDA':
        all_with_track_data=['01_University_Canteen','04_Primary_School','05_Basketball_Court', '06_Xinzhongguan', '10_Huaqiangbei','07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '02_OCT_Habour', '03_Xili_Crossroad']

        train_test_split={
            'Group':
                {'stage1':{'train':['04_Primary_School','05_Basketball_Court', '06_Xinzhongguan', '10_Huaqiangbei','07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '02_OCT_Habour', '03_Xili_Crossroad'],
                      },#'02_OCT_Habour', '03_Xili_Crossroad', '04_Primary_School', '05_Basketball_Court', '06_Xinzhongguan', '07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '10_Huaqiangbei'
                'distill':{'train':['04_Primary_School','05_Basketball_Court', '06_Xinzhongguan', '10_Huaqiangbei','07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '02_OCT_Habour', '03_Xili_Crossroad']},
                'stage2':{'train':
                    ['04_Primary_School','05_Basketball_Court', '06_Xinzhongguan', '10_Huaqiangbei','07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '02_OCT_Habour', '03_Xili_Crossroad'], 'test':['01_University_Canteen']}}, #['02_OCT_Habour', '03_Xili_Crossroad', '04_Primary_School', '05_Basketball_Court', '06_Xinzhongguan', '07_University_Campus', '08_Xili_Street_1', '09_Xili_Street_2', '10_Huaqiangbei']
            'Interaction':
                {'stage1':{'train':['01_University_Canteen','02_OCT_Habour', '03_Xili_Crossroad'],
                      },
                'distill':{'train':['01_University_Canteen','02_OCT_Habour', '03_Xili_Crossroad']},
                'stage2':{'train':['01_University_Canteen','02_OCT_Habour', '03_Xili_Crossroad'],'test':['07_University_Campus']}} #, , '08_Xili_Street_1', '09_Xili_Street_2', '10_Huaqiangbei'
        }

        if cfg.training_stage==1: # This will auto include the distill stage.
            if cfg.distill:
                training_set = PANDADataset(cfg,train_test_split[cfg.core_task]['stage1']['train'],augment='distill')
                validation_set = PANDADataset(cfg,train_test_split[cfg.core_task]['distill']['train'],augment='distill') # It's distill set instead, but for convinience.
            else:
                training_set = PANDADataset(cfg,train_test_split[cfg.core_task]['stage1']['train'],augment='train')
                validation_set = PANDADataset(cfg,train_test_split[cfg.core_task]['distill']['train'],augment='distill') # It's distill set instead, but for convinience.
            
        else:
            train_videos=train_test_split[cfg.core_task]['stage2']['train']
            test_videos=train_test_split[cfg.core_task]['stage2']['test']
            training_set = PANDADataset(cfg,train_videos,augment='none')
            validation_set = PANDADataset(cfg,test_videos,augment='none')

    else:
        assert False
                                         
    
    print('Reading dataset finished...')
    print('%d train samples'%len(training_set))
    print('%d test samples'%len(validation_set))
    
    return training_set, validation_set
    