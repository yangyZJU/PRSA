from abc import ABC
import llm
import re

class PromptOptimizer(ABC):
    def __init__(self, args, scorer, gradient_dict):
        self.opt = args
        self.scorer = scorer
        self.gradient_dict = gradient_dict

class PAA(PromptOptimizer):
    """ PAA: Prompt Attention Algorithm
        This idea of Prompt Attention Algorithm is borrowed from this paper: "Automatic Prompt Optimization with "Gradient Descent" and Beam Search"
    """
    

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            texts.append(text[start_index:end_index].strip())
            text = text[end_index+len(end_tag):]
        return texts

    def filter_target_score(self, text, feedbacks):
        if feedbacks == []:
            numbers = re.findall(r'\d+\.\d+|\d+', text)
            target_score = [float(number) for number in numbers if float(number) < 10]
            assert len(target_score) == 1
        else:
            target_score = feedbacks
        return target_score[0]


    def extract_number(self, s):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        return numbers[0] if numbers else None

    def cal_gradients(self, generated_output, output_data):

        gradient = {}
        if self.opt["theme"] in ["Music", "Sports"]:
            self.opt["attention_threshold"] = 8

        elements = ['Characteristic','Topic','Argument','Structure','Style','Tone','Purpose','Sentence Type','Audience','Background']
        for idx, element in enumerate(elements):
            gradient_prompt = f"""
            Generated Output:
            "{generated_output}"

            Real Output:
            "{output_data}"

            Score based on {element} similarity between Generated Output and Real Output, if full score is 10.
            The score is wrapped with <START> and <END>
            """
            gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
            res = llm.chatGPT(gradient_prompt, model="gpt-3.5-turbo", temperature=0.0)
            feedbacks = []
            temp = []
            for r in res:    
                temp += self.parse_tagged_text(r, "<START>", "<END>")
                feedbacks = self.filter_target_score(r, temp)
            try:
                if float(feedbacks) < self.opt["attention_threshold"]:
                    print("feedback socre: ",feedbacks)
                    gradient[element] = 1
            except:
                if float(self.extract_number(feedbacks)) < self.opt["attention_threshold"]:
                    print("extract feedback score: ",float(self.extract_number(feedbacks)))
                    gradient[element] = 1

        return gradient


    

        


    

    