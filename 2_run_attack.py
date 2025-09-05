
import os
from tqdm import tqdm
import json
import argparse
import pandas as pd
import math
import dataset
import utils
import llm
import scorers
from sentence_bert import calculate_similarity_sbert



def is_valid_score(x):
    return x != 0 and not math.isinf(x) and not math.isnan(x)


def save_to_csv(df, directory, filename):
    """Save DataFrame to a CSV file, ensuring the directory exists."""
    os.makedirs(directory, exist_ok=True)
    csv_file = os.path.join(directory, filename)
    df.to_csv(csv_file, index=False)
    print(f"File saved: {csv_file}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--theme', default='demo_data')
    parser.add_argument('--data_dir', default='demo_data')
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--beam_search', default=1, type=int)
    parser.add_argument('--pre_pruning', default=1, type=int)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--related_words_interval', default=5, type=int)
    parser.add_argument('--out', default='log/test_out.txt')
    parser.add_argument('--temperature', default=0.7, type=float)

    parser.add_argument('--evaluator', default="syntactic_similarity", type=str)
    parser.add_argument('--m', default=3, type=int, help='m is the sampling number for each user input')
    parser.add_argument('--n', default=2, type=int, help='n is the number of the test user input')
    parser.add_argument('--scorer', default="syntactic_similarity", type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = get_args()
    config = vars(args)

    data = dataset.Datasets(config)
    scorer = scorers.MetricsScorer(args.evaluator, args.m)
    model = llm.ChatGPTPredictor(config)
    test_data = data.load_test_data()

    if os.path.exists(args.out):
        os.remove(args.out)
    print(config)

    scores_list = []
    save_file_name = config["theme"] + "_res"
    llm_based_eva_score_list = []
    for idx, batch in tqdm(enumerate(test_data)):
        input_data = batch['Preview Input']
        print(idx)
        
        category = batch['Category']
        print("\nGet the prompt category:", category)
        config["theme"] = category

        gradient_path = "model/gradient_" + config["theme"] + ".json"
        with open(gradient_path, 'r') as file:
            gradient_dict = json.load(file)
        
        print("\n========Start Stealing Target Prompt========  ... ...")

        target_prompt = batch['Prompt']
        output_data = batch['Preview Output']
        input_case1 = batch['Input Test Case1']
        input_case2 = batch['Input Test Case2']
        inputs = [input_case1, input_case2]
        inputs = inputs[:args.n]
        
        stolen_prompt = llm.generate_prompt(config, input_data, output_data, gradient_dict)
        stolen_prompt = utils.prompt_pruning_google(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt)

        print("\nStolen prompt is :", stolen_prompt)

        '''
        (Optional 1)
        stolen_prompt = utils.prompt_pruning(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt_1)

        (Optional 2)
        stolen_prompt = utils.prompt_pruning_phrase_level(config, stolen_prompt, input_data, output_data, model.inference)
        stolen_prompt = utils.format_clean(stolen_prompt)
        '''

        with open(args.out, 'a') as outf:
            outf.write(json.dumps(f"input_data: {input_data}") + '\n')
            outf.write(json.dumps(f"stolen_prompt: {stolen_prompt}") + '\n')
            outf.write(json.dumps(f"target_prompt: {target_prompt}") + '\n')

        print("\n========Evaluation of Prompt Similarity======== ... ...")
        prompt_sim_score = calculate_similarity_sbert(target_prompt, stolen_prompt)
        print("\nPrompt similarity score is :", prompt_sim_score)


        print("\n========Evaluation of Functional Consistency======== ... ...")
        target_output_sim_score = scorer.evaluate_target_prompt(model.inference, inputs, target_prompt)
        if target_output_sim_score is None:
            print(f"[Warning] Skipped iteration {idx} due to failed target output generation.")
            continue

        stolen_output_sim_score = scorer.evaluate_stolen_prompt(model.inference, inputs, target_prompt, stolen_prompt)

        semantic_target_score,  syntactic_target_score, js_target_score = target_output_sim_score
        semantic_stolen_score, syntactic_stolen_score, js_stolen_score = stolen_output_sim_score

        #We normalize our results against the similarity score between target outputs, producing a similarity score in the range (0,1)
        if all(map(is_valid_score, [semantic_target_score, syntactic_target_score, js_target_score])):
            semantic_sim_socre = min(semantic_stolen_score / semantic_target_score, 1)
            syntactic_sim_socre = min(syntactic_stolen_score / syntactic_target_score, 1)
            js_sim_socre = min(js_stolen_score / js_target_score, 1)

            print("\nSemantic similarity score is:", semantic_sim_socre)
            print("\nSyntactic similarity score is:", syntactic_sim_socre)
            print("\nStructural similarity score is:", js_sim_socre)
            with open(args.out, 'a') as outf:
                outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] semantic similarity score: {semantic_sim_socre:.4f}") + '\n')
                outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] syntactic similarity score: {syntactic_sim_socre:.4f}") + '\n')
                outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] structural similarity score: {js_sim_socre:.4f}") + '\n')
                outf.write(json.dumps(f"Round [{idx+1}/{len(test_data)}] prompt similarity score: {prompt_sim_score:.4f}") + '\n')
                outf.write('\n\n')

        
            scores_list.append({
                'iteration': idx,
                'target prompt': target_prompt,
                "stolen prompt": stolen_prompt,
                'semantic similarity score': semantic_sim_socre,
                'syntactic similarity score': syntactic_sim_socre,
                'structural similarity score': js_sim_socre,
                'prompt similarity score': prompt_sim_score
            })
        else:
            print(f"[Warning] Skipped iteration {idx} due to invalid denominator (0/inf/NaN) in target scores.")

        
        
        print("\n======== LLM-based Multi-dimensional Evaluation======== ... ...")
        for input_case in inputs:
            target_output = model.inference(input_case, target_prompt)
            generated_output =  model.inference(input_case, stolen_prompt)
            
            if target_output is None or generated_output is None:
                print(f"[Warning] Skipped input '{input_case}' due to inference failure.")
                continue

            llm_eva_res = llm.llm_based_evaluation(target_output, generated_output)
            try:
                llm_eva_scores = json.loads(llm_eva_res)
                print("LLM-based evaluation score: ", llm_eva_scores)
            except json.JSONDecodeError:
                llm_eva_scores = {"Accuracy":0, "Completeness":0, "Tone":0, "Sentiment":0, "Semantics":0}
                print("Error: Could not parse response as JSON.")
                print("Raw response:", llm_eva_res)

            llm_based_eva_score_list.append((input_case, target_prompt, stolen_prompt, 
                               llm_eva_scores["Accuracy"], llm_eva_scores["Completeness"],llm_eva_scores["Tone"],llm_eva_scores["Sentiment"],llm_eva_scores["Semantics"]))

    
    directory = "result"
    df = pd.DataFrame(llm_based_eva_score_list, columns=['user input', 'target prompt', 'stolen prompt',
                                        'accuracy', 'completeness', 'tone', 'sentiment', 'semantics'])
    save_to_csv(df, directory, f"llm_based_eva_{save_file_name}.csv")

    df_scores = pd.DataFrame(scores_list)
    save_to_csv(df_scores, directory, f"{save_file_name}.csv")

