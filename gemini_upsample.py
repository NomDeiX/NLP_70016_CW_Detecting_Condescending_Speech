import google.generativeai as genai
import pandas as pd
from tqdm.notebook import tqdm
import random

class GeminiTextUpsampler:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model_name = "gemini-1.5-pro"
    
    def get_sentence_variation(self, sentence, label):
        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""For the following sentence, return a short sentence that will have similar writing style, but use different words and describe a different scenario.
            The output should have similar emotional charge as the input sentence. If the sentence is labelled as 1 (condescending) it should have the same subtle features of patronizing and condescending language.
            > Sentence: '{sentence}'
            > Label: {label}
            Return nothing but a single sentence with a single dot at the end
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def get_sentence_variation_few_examples(self, sentence, label):
        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""For the following sentence, return a short sentence that will have similar writing style, but use different words and describe a different scenario.
            The output should have similar emotional charge as the input sentence. If the sentence is labelled as 1 (condescending) it should have the same subtle features of patronizing and condescending language.

            Example of a new generation is for this positive sample: 'Fast food employee who fed disabled man becomes internet sensation' into something like 'Cashier who helped elderly woman went viral online.'

            Examples of negative (labelled 0) are:
            1) Here 's a woman apparently hopelessly in love , shot several times by her lover , the owner of a mansion in a gated community full of millionaires just like him 
            2) It just seems hopeless
            3) The H1-B visa is a non-immigrant visa that allows US companies to employ foreign workers in speciality occupations that require theoretical or technical expertise .
            4) An intellectually disabled woman has moved out of her house in fear after a Wellington man forced his way into her home and threatened her with a Taser for fun 

            Examples of positive (labelled 1) are:
            1) Guinness World Record of 540lbs of 7-layer munchy monstrosity , he tweeted . Delivered immediately to LA 's Midnight Mission for the homeless . A good day . 
            2) She has one huge platform , and information can go out to places that really need to know what 's going on in her hometown . She has always availed not only her platform , her voice , but also resources -- tangible resources -- to help those most in need in Houston and around 
            3) B.C. housing minister Selina Robinson said that the project represents hope for many women and families in need .
            4) Thankfully , while Krishna Tulasi ca n't entirely escape from the trope of disabled persons with hearts of gold , it manages to do better than many previous films with disabled protagonists

            Your sentence and label are:
            > Sentence: '{sentence}'
            > Label: {label}
            Return nothing but a single sentence with a single dot at the end
            """
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
    
    def gemini_upsample(self, df, new_generations_per_sentence):
        new_rows = []
        df_len = len(df)
        iterator = tqdm(df.iterrows(), total=df_len, desc="Generating variations") if df_len > 100 else df.iterrows()

        for _, row in iterator:
            for _ in range(new_generations_per_sentence):
                variation = self.get_sentence_variation(row['text'], row['label'])
                if variation:
                    new_rows.append({'text': variation, 'label': row['label']})

        return pd.DataFrame(new_rows, columns=['text', 'label'])
    
    def gemini_upsample_positives(self, df, new_generations_per_positive=3, negative_sampling_rate=0.33):
        new_rows = []
        df_len = len(df)
        iterator = tqdm(df.iterrows(), total=df_len, desc="Generating variations") if df_len > 100 else df.iterrows()

        for _, row in iterator:
            variations_to_generate = new_generations_per_positive if row['label'] == 1 else 1 if random.random() <= negative_sampling_rate else 0
            
            for _ in range(variations_to_generate):
                variation = self.get_sentence_variation(row['text'], row['label'])
                if variation:
                    new_rows.append({'text': variation, 'label': row['label']})

        return pd.DataFrame(new_rows, columns=['text', 'label'])

# Example usage:
# api_key = "REMOVED FROM THE SUBMISSION"
# upsampler = GeminiTextUpsampler(api_key)
# upsampled_df = upsampler.gemini_upsample_positives(data, new_generations_per_positive=3, negative_sampling_rate=0.1)
# upsampled_df.to_csv('gemini_upsampled_data.tsv', sep='\t', index=False)
