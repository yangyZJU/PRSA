import pickle
import os
from tqdm import tqdm
import json
import argparse
import scorers
import dataset
import llm
import optimizers
import utils



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default='Email', help='theme is categpry of the prompts.')
    parser.add_argument('--data_dir', default='collect_data')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--out', default='test_out.txt')
    parser.add_argument('--attention_threshold', default=7.5, type=float, help='attention_threshold needs to update based on different gpt model inference.')

    parser.add_argument('--evaluator', default="semantic_similarity", type=str)
    parser.add_argument('--m', default=3, type=int, help='m is the sampling number for each user input.')
    parser.add_argument('--scorer', default="semantic_similarity", type=str)
    
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    config = vars(args)

    data = dataset.Datasets(config)
    scorer = scorers.MetricsScorer(args.evaluator, args.m)
    model = llm.ChatGPTPredictor(config)
    optimizer = optimizers.PAA(
        config, scorer, {})


    if os.path.exists(args.out):
        os.remove(args.out)

    print(config)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(config) + '\n')
    
    collect_data = data.load_collect_data()
    num_collect_data = len(collect_data)
    gradient_dict={}
    for epoch in range(args.epochs):
        for idx, batch in tqdm(enumerate(collect_data)):
            input_data = batch['Preview Input']
            target_prompt = batch['Prompt']
            output_data = model.inference(input_data, target_prompt)

            if gradient_dict == {}:
                base_stolen_prompt = llm.generate_prompt(config, input_data, output_data, gpt_model="gpt-3.5-turbo-1106")
                
            else:
                base_stolen_prompt = llm.generate_prompt(config, input_data, output_data, gradient_dict, gpt_model="gpt-3.5-turbo-1106")
            
            if base_stolen_prompt == None:
                    continue
            generated_output = model.inference(input_data, base_stolen_prompt)
            gradient = optimizer.cal_gradients(generated_output, output_data)
            print("gradient: ", gradient)
            utils.update_gradient_dict(gradient, gradient_dict)     

    final_gradient = utils.filter_and_sort_dict(gradient_dict, num_collect_data/10)

    with open(args.out, 'a') as outf:
        outf.write(json.dumps(f"{args.theme}: {final_gradient}") + '\n')

    with open(f'model/gradient_{args.theme}.pkl', 'wb') as f:
        pickle.dump(final_gradient, f)

    with open(f'model/gradient_{args.theme}.json', 'w') as json_file:
        json.dump(final_gradient, json_file)
