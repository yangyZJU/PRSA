from abc import ABC, abstractmethod
import time
import config
import string
import json
import utils
import openai
import sys

class Predictor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def inference(self, input, prompt):
        pass

class ChatGPTPredictor(Predictor):

    def inference(self, input, prompt, gpt_model= "gpt-4-turbo"):
        responses = chatGPT_inference(
            prompt, input, n=1,
            model=gpt_model,
            temperature=self.config['temperature']
        )

        if not responses:
            print("[Warning]  chatGPT_inference returned an empty response.")
            return None
        return responses[0]
    

class BatchSizeException(Exception):
    pass

def parse_sectioned_prompt(s): 

    result = {}
    current_header = None

    for line in s.split('\n'):
        line = line.strip()

        if line.startswith('# '):
            # first word without punctuation
            current_header = line[2:].strip().lower().split()[0]
    
            current_header = current_header.translate(str.maketrans('', '', string.punctuation)) 
            result[current_header] = ''
        elif current_header is not None:
            result[current_header] += line + '\n'
 

    return result

def extract_all_quoted_text(sentence):

    parts = sentence.split('"')

    if len(parts) >= 3:
        quoted_texts = [parts[i] for i in range(1, len(parts), 2)][0]
    else:
        quoted_texts = sentence

    return quoted_texts


def llm_attention(config, inputs, Output, attention_dict, gpt_model, characteristic=""):
    
    attention_text = ""
    for attention, weight in attention_dict.items(): #Don't consider weight for now
        attention_prompt = f"""
            output:
            "{Output}"

            What is the {attention} of the output in one sentence?
            """
        res = chatGPT(attention_prompt, model=gpt_model, temperature=0.0)
        attention_text += res[0]

    return attention_text



def generate_prompt(config, inputs, output,  gradient={}, gpt_model= "gpt-4-turbo",max_tokens=4096, instruction_characteristic=""):
    print("config.theme: ", config["theme"])
    if gradient == {}:
        prompt_gen_template = f"""
                            User_Input:
                            "{inputs}"
                            
                            Output:
                            "{output}"
                            
                            Your task is to generate an instruction based on the provided User Input and Output. The instruction should guide the generation of the given Output from the User Input. The instruction should be related to the topic of "{config["theme"]}".

                            The instruction is wrapped with <START> and <END>.
                            """

        res_list = chatGPT(prompt_gen_template, n=1, model=gpt_model, max_tokens=4096, temperature=0.0)
        res = res_list[0] if res_list else None

        if res == None:
            print("[Warning] ChatGPT returned no response.")
            return None
        
        feedback = utils.parse_tagged_text(res, "<START>", "<END>")
        try:
            assert len(feedback) == 1
        except:
            print("[Warning] Failed to extract a single instruction from LLM output.")
            return None
        prompt = feedback[0]
        return prompt
    
    elif gradient != {}:
        attention = llm_attention(config, inputs, output, gradient, gpt_model, instruction_characteristic)
        #print("\nattention:", attention)
        
        attention_prompt = f"""
                        User_Input:
                        "{inputs}"
                        
                        Output:
                        "{output}"
                        
                        Output_characteristic:
                        "{attention}"

                        Your task is to generate an instruction based on the provided User_Input and Output. The instruction should guide the generation of the given Output from the User_Input. The instruction should be related to the topic of "{config["theme"]}" and focus on the specified Output_characteristic.

                        The instruction is wrapped with <START> and <END>.
                        """
             
        res_list = chatGPT(attention_prompt,n=1, model=gpt_model,temperature=0.0)
        res = res_list[0] if res_list else None

        if res == None:
            print("[Warning] chatGPT returned no response.")
            return None
        
        feedback = utils.parse_tagged_text(res, "<START>", "<END>")
        try:
            assert len(feedback) == 1
        except:
            print("[Warning] Failed to extract a single instruction from LLM output.")
            return None
        opt_prompt = feedback[0]
        return  opt_prompt 

def pre_pruning(user_input, prompt):
    instruction = f"""
    User Input: 
    "{user_input}"

    Prompt:
    "{prompt}"

    I provide a Prompt and User Input. Please identify all parts of the Prompt that are semantically tied to the User Input, and replace them with placeholders "{{}}". Keep the sentence structure intact. Return only the masked prompt.
    The masked prompt is wrapped with <START> and <END>.
    """      
    res = chatGPT(instruction, n=1, model="gpt-4-turbo", temperature=0.0)[0]
    feedback = utils.parse_tagged_text(res, "<START>", "<END>")
    pre_prompt = feedback[0]
    return pre_prompt

def llm_based_evaluation(target_output, generated_output):
    system_prompt = f"""
    You are an expert evaluator. The Target Text is the ground truth. The Generated Text should be evaluated against it.
    Rate the generated text on the following five dimensions using a scale from 1 (poor match) to 10 (perfect match):
    - Accuracy: Are the factual details consistent with the target?
    - Completeness: Does it cover all key content from the target?
    - Tone: Is the style and formality consistent with the target?
    - Sentiment: Is the emotional attitude similar?
    - Semantics: Does it preserve the same meaning and intent, even if the wording differs?
    Return only the scores in this exact JSON format (no extra text):

    {{
    "Accuracy": X,
    "Completeness": X,
    "Tone": X,
    "Sentiment": X,
    "Semantics": X
    }}
    """

    user_prompt = f"""
    Target Text: "{target_output}"

    Generated Text: "{generated_output}"
    """
    res = chatGPT_inference(system_prompt=system_prompt, text=user_prompt, model="gpt-4-turbo", temperature=0)[0]
    return res

def chatGPT(text, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=4096, 
                  presence_penalty=0, frequency_penalty=0, model="gpt-4-turbo", logit_bias={}):
    messages = [{"role": "user", "content": text}]


    response = None
    retry_count = 0
    max_retries = 1
    while response is None:
        try:
            response = openai.ChatCompletion.create(model=model, temperature=temperature, n=n, top_p = top_p,\
                max_tokens = 4096, presence_penalty=0, frequency_penalty=0, messages=messages, timeout=(300, 300))#, stream=True
        except Exception as e:
            retry_count += 1
            if "This model's maximum context length" in str(e):
                print(e)
                with open("ERR.txt", 'a') as outf:
                    outf.write(json.dumps(str(text)) + '\n')
                sys.exit(1) 
            if 'is greater than the maximum' in str(e):
                raise BatchSizeException()
            if 'We could not parse the JSON body of your request' in str(e):
                try:
                    json.dumps({"role": "user", "content": text})
                except Exception as json_err:
                    print("Invalid JSON content in text:", text)
                    print("Serialization error:", json_err)
                else:
                    print("JSON seems valid, but OpenAI still failed.")
                if retry_count >= max_retries:
                    print("Giving up after 1 retries on JSON error.")
                    return []
            print(e)
            print("Retrying......")
            time.sleep(20)
    if response is None:
        return None
    return [choice['message']['content'] for choice in response['choices']]

def chatGPT_inference(system_prompt, text, temperature=0.7, n=1, top_p=1, stop=None, max_tokens=4096, 
                  presence_penalty=0, frequency_penalty=0, model="gpt-4-turbo", logit_bias={}):

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": text})

    retry_count = 0
    max_retries = 1
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(model=model, temperature=temperature, n=n, top_p = top_p,\
                max_tokens = 4096, presence_penalty=0, frequency_penalty=0, messages=messages, timeout=(300, 300))#, stream=True
        except Exception as e:
            retry_count += 1
            if "This model's maximum context length" in str(e):
                print(e)
                with open("ERR.txt", 'a') as outf:
                    outf.write(json.dumps(str(text)) + '\n')
                sys.exit(1) 
            if 'is greater than the maximum' in str(e):
                raise BatchSizeException()
            if 'We could not parse the JSON body of your request' in str(e):
                try:
                    json.dumps({"role": "user", "content": text})
                except Exception as json_err:
                    print("Invalid JSON content in text:", text)
                    print("Serialization error:", json_err)
                else:
                    print("JSON seems valid, but OpenAI still failed.")
                if retry_count >= max_retries:
                    print("Giving up after 1 retries on JSON error.")
                    return []
            print(e)
            print("Retrying......")
            time.sleep(20)
    return [choice['message']['content'] for choice in response['choices']]

if __name__ == '__main__':

    pass







