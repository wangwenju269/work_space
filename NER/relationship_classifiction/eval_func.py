import json
from utils.arguments_parse import args
from matplotlib import pyplot as plt

def load_data(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences = []
        for line in lines:
            data = json.loads(line)
            args_list=[]
            for event in data['event_list']:
                event_type = event['event_type']
                args=[event_type+'_'+arg['argument'] for arg in event['arguments']]
                args_list.extend(args)
            sentences.append(args_list)
                    
        return sentences


def eval_function():

    true_data_list=load_data(args.test_path)
    pred_data_list=load_data('./output/result2.json')

    true_count=0
    pred_count=0
    corr_count=0

    for i in range(1498):
        true_data = true_data_list[i]
        pred_data = pred_data_list[i]
        corr = sum([1 for k in true_data if k in pred_data])

        true_count += len(true_data)
        pred_count += len(pred_data)
        corr_count += corr
    
    recall = corr_count / true_count
    precise = corr_count / pred_count

    print(recall)
    print(precise)


def save_s():

    with open("./output/duie_w.json", 'r', encoding='utf8') as f:
        lines = f.readlines()
        count=0
        sentences = []
        for line in lines:
            data = json.loads(line)
            for i in range(len(data['spo_list'])):
                if 'object' not in data['spo_list'][i].keys():
                    del data['spo_list'][i]
                if '@value' not in data['spo_list'][i]['object'].keys():
                    data['spo_list'][i]['object']['@value']='flag'
                if '@value' not in data['spo_list'][i]['object_type'].keys():
                    data['spo_list'][i]['object_type']['@value']='人物'
            sentences.append(data)
            count+=1
    sentences=sentences[:-1]
    with open("./output/duie_2.json", 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            for i in range(len(data['spo_list'])):
                if 'object' not in data['spo_list'][i].keys():
                    del data['spo_list'][i]
                if '@value' not in data['spo_list'][i]['object'].keys():
                    data['spo_list'][i]['object']['@value']='flag'
                if '@value' not in data['spo_list'][i]['object_type'].keys():
                    data['spo_list'][i]['object_type']['@value']='人物'
            sentences.append(data)
            count+=1
    
    # with open("./data/duie_test1.json", 'r', encoding='utf8') as f:
    #     lines = f.readlines()

    #     for i,line in enumerate(lines):
    #         if i>=count:
    #             data = json.loads(line)
    #             data['spo_list']=[]
    #             sentences.append(data)

    with open("./output/duie.json", 'w', encoding='utf8') as f:
        for s in sentences:
            tmp=json.dumps(s,ensure_ascii=False)
            f.write(tmp+'\n')

def analysis_data():
    with open("./data/duie_train.json", 'r', encoding='utf8') as f:
        lines = f.readlines()
        sentences=[]
        for i,line in enumerate(lines):
            data = json.loads(line)
            sentences.append(len(data['text']))

    plt.plot(sentences)
    plt.show()



if __name__=='__main__':
    # save_s()
    analysis_data()
            