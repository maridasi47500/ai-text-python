from country import Country
from urllib.request import Request, urlopen
from chercherimage import Chercherimage
from bs4 import BeautifulSoup
from fichier import Fichier
from pics import Pics
from somescaffold import Scaffold
import scacy
import json
import numpy as np
from sklearn.model_selection import train_test_split
class Addscript():
      def __init__(self):
            self.paysdb=Country()
            self.picdb=Pics()
            self.URL = "https://www.enchantedlearning.com/wordlist/jobs.shtml"
            self.req = Request(self.URL , headers={'User-Agent': 'Mozilla/5.0'})
            self.webpage = urlopen(self.req).read()
            self.soup = BeautifulSoup(self.webpage, "html.parser")
      def hello(self):
            Fichier("./uploads","transformer1.sh").ecrire("pip3 install transformers datasets evaluate accelerate")
            hey={"imdb":[{
                        "label": 0,
                            "text": "I love sci-fi and am willing to put up with a lot. Sci-fi movies/TV are usually underfunded, under-appreciated and misunderstood. I tried to like this, I really did, but it is to good TV sci-fi as Babylon 5 is to Star Trek (the original). Silly prosthetics, cheap cardboard sets, stilted dialogues, CG that doesn't match the background, and painfully one-dimensional characters cannot be overcome with a 'sci-fi' setting. (I'm sure there are those of you out there who think Babylon 5 is good sci-fi TV. It's not. It's clichéd and uninspiring.) While US viewers might like emotion and character development, sci-fi is a genre that does not take itself seriously (cf. Star Trek). It may treat important issues, yet not as a serious philosophy. It's really difficult to care about the characters here as they are not simply foolish, just missing a spark of life. Their actions and reactions are wooden and predictable, often painful to watch. The makers of Earth KNOW it's rubbish as they have to always say \"Gene Roddenberry's Earth...\" otherwise people would not continue watching. Roddenberry's ashes must be turning in their orbit as this dull, cheap, poorly edited (watching it without advert breaks really brings this home) trudging Trabant of a show lumbers into space. Spoiler. So, kill off a main character. And then bring him back as another actor. Jeeez! Dallas all over again.",
                            }]}
            Fichier("./uploads","transformer1.json").ecrirejson(hey)
            transformer1="""from transformers import AutoTokenizer
from fichier import Fichier
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
class Transformer1():
    def __init__(self,modelname="distilbert/distilbert-base-uncased"):
        self.modelname=modelname
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.imdb=Fichier("./welcome","transformer1.json").lirejson()
    def preprocess_function(self,exemples):
        return tokenizer(exemples["text"], truncation=True)
    def tokenize_imdb(self):
        self.tokenized_imdb = self.imdb.map(self.preprocess_function, batched=True)
    def create_examples(self,exemples):
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    def load_accuracy_metric(self):
        self.accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.accuracy.compute(predictions=predictions, references=labels)
    def create_a_map(self):
        self.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
        self.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
    def train():
        model = AutoModelForSequenceClassification.from_pretrained(
        self.modelname, num_labels=2, id2label=self.id2label, label2id=self.label2id
        )
        training_args = TrainingArguments(
            output_dir="my_awesome_model",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_imdb["train"],
            eval_dataset=self.tokenized_imdb["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
    def inference(self,text):
        classifier = pipeline("sentiment-analysis", model="./my_awesome_model")
        return classifier(text)
myai=Transformer1()
myai.tokenize_imdb()
myai.create_examples()
myai.load_accuracy_metric()
myai.create_a_map()
myai.train()
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."
print(myai.inference(text))
            """
            Fichier("./","transformer1.py").ecrire(transformer1)
            Fichier("./uploads","transformer2.sh").ecrire("pip3 install transformers datasets evaluate seqeval")
            text1=""
            nlp=spacy.load("en_core_web_sm")
            mysentence=("Microsoft Corporation is a multinational technology with headquarters in Redmond")
            wnut=[]
            doc=nlp(mysentence)
            tokens=mysentence.replace("'"," '").split(" ")
            featurenames=[]
            mystr=""
            i=0
            for x in tokens:
                begin1=False
                inside1=False
                mystr=""
                for ent in doc.ents:
                   print(ent.text+" -- "+ent.label_+spacy.explain(ent.label_)+"--"+ent.start_char+"--"+ent.end_char+"--")
                   if i == ent.start_char:
                       mystr+="B-"+ent.label_
                       begin1=True
                       continue
                   elif i < ent.start_char && i <= ent.end_char:
                       mystr+="-"+ent.label_
                       inside1=True
                       continue
                if not begin1 and not inside1:
                   mystr="0"
                featurenames.append(mystr)
                i+=1
            wnut1["ner_tags"]=featurenames
            wnut.append(wnut1)
            Fichier("./uploads","transformer2.json").ecrirejson({"list":wnut})
            transformer2= """from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import pipeline
class Transformer2():
    def __init__(self,modelname="distilbert/distilbert-base-uncased"):
        self.modelname=modelname
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.wnut=Fichier("./welcome","transformer2.json").lirejson()["list"]
        self.exemple=Fichier("./welcome","transformer2.json").lirejson()["list"][0]
        self.tokenized_input = self.tokenizer(self.example["tokens"], is_split_into_words=True)
        self.labels = [label_list[i] for i in self.exemple[f"ner_tags"]]
        self.tokens= self.tokenizer.convert_ids_to_tokens(self.tokenized_input["input_ids"])
    def tokenize_and_align_labels(self,examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
        labels = []
        for i, label in enumerate(self.exemple["ner_tags"]):
            word_ids = self.tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    def preprocess_function(self):
        self.tokenized_wnut = self.wnut.map(self.tokenize_and_align_labels, batched=True)
    def create_example(self):
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
    def load_method(self):
        self.seqeval = evaluate.load("seqeval")
    def compute_metrics(self,p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    def create_map(self):
        self.id2label=enumerate(self.exemple["ner_tags"])
        self.label2id= {v: k for k, v in self.id2label.items()}
    def train(self):
        model = AutoModelForTokenClassification.from_pretrained(
            self.modelname, num_labels=13, id2label=self.id2label, label2id=self.label2id
        )
        training_args = TrainingArguments(
        output_dir="my_awesome_wnut_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        )
        trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=self.tokenized_wnut["train"],
        eval_dataset=self.tokenized_wnut["test"],
        tokenizer=self.tokenizer,
        data_collator=self.data_collator,
        compute_metrics=self.compute_metrics,
        )
        trainer.train()
    def inference(self,text):
        classifier = pipeline("ner", model="my_awesome_wnut_model")
        return classifier(text)
ai = Transformer2()
ai.preprocess_function()
ai.load_method()
ai.create_example()
ai.create_map()
ai.train()
text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
print(ai.inference(text))
        """
            Fichier("./","transformer2.py").ecrire(transformer2)
            Fichier("./uploads","transformer3.sh").ecrire("pip3 install transformers datasets evaluate ")
            x=y=[{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
             'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
              'id': '5733be284776f41900661182',
               'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
                'title': 'University_of_Notre_Dame'
                }]
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.4, random_state=0
            )
            squad={"train":x_train, "test":x_test}
            Fichier("./uploads","transformer3.json").ecrirejson(squad)
            transformer3="""from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
import evaluate
import numpy as np
from transformers import pipeline
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
class Transformer3():
    def __init__(self,modelname="distilbert/distilbert-base-uncased"):
        self.modelname=modelname
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.squad=Fichier("./welcome","transformer3.json").lirejson()
        
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []
        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1
            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs
    def apply_preprocess_function(self):
        self.tokenized_squad = self.squad.map(self.preprocess_function, batched=True, remove_columns=self.squad["train"][0].keys())
    def create_examples(self):
        self.data_collator = DefaultDataCollator()
    def train(self):
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.modelname)
        training_args = TrainingArguments(
            output_dir="my_awesome_qa_model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=True,
        )
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.tokenized_squad["train"],
            eval_dataset=self.tokenized_squad["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
        )
        trainer.train()
    def inference(self,question,context):
        question_answerer = pipeline("question-answering", model="my_awesome_qa_model")
        return question_answerer(question=question, context=context)
ai=Transformer3()
ai.apply_preprocess_function()
ai.create_examples()
ai.train()
question = "How many programming languages does BLOOM support?"
context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
print(ai.inference(question,context))
        """
        Fichier("./","transformer3.py").ecrire(transformer3)
        ask=[{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers': {'a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
  'text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
   'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
   'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
   'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
  'score': [21, 19, 5, 3],
  'text_urls': [[],
   [],
   [],
   ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
 'title_urls': ['url'],
 'selftext_urls': ['url']}]
        Fichier("./uploads","transformer4.sh").ecrire("pip3 install transformers datasets evaluate")
        x_train, x_test, y_train, y_test = train_test_split(
            ask, ask, test_size=0.4, random_state=0
        )
        squad={"train":x_train, "test":x_test}
        Fichier("./uploads","transformer4.json").ecrirejson(squad)
        transformer4="""from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import evaluate
import numpy as np
from transformers import pipeline
from transformers import DefaultDataCollator
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import math
class Transformer4():
    def __init__(self,modelname="distilbert/distilgpt2"):
        self.modelname=modelname
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.eli5=Fichier("./welcome","transformer4.json").lirejson()
        self.eli5=self.eli5.flatten()
        self.block_size = 128
    def preprocess_function(self,examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])
    def apply_preprocess_function(self):
        self.tokenized_eli5 = self.eli5.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=self.eli5["train"][0].keys(),
        )
    def group_texts(self,examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    def apply_group_texts_function(self,examples):
        self.lm_dataset = self.tokenized_eli5.map(self.group_texts, batched=True, num_proc=4)
    def create_examples(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
    def train(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.modelname)
        training_args = TrainingArguments(
            output_dir="my_awesome_eli5_clm-model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            push_to_hub=True,
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["test"],
            data_collator=self.data_collator,
        )
        self.trainer.train()
    def evaluate(self):
        eval_results = self.trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    def inference(self,prompt):
        generator = pipeline("text-generation", model="my_awesome_eli5_clm-model")
        return generator(prompt)
ai=Transformer4() 
ai.apply_preprocess_function()
ai.apply_group_texts_function()
ai.create_examples()
ai.train()
ai.evaluate()
prompt = "Somatic hypermutation allows the immune system to"
print(ai.inference(prompt)
    
        """
        Fichier("./","transformer4.py").ecrire(transformer4)
        transformer5="""from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import evaluate
import numpy as np
from transformers import AutoModelForMaskedLM
from transformers import pipeline
from transformers import DefaultDataCollator
import math
class Transformer5():
    def __init__(self,modelname="distilbert/distilgpt2"):
        self.modelname=modelname
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelname)
        self.eli5=Fichier("./welcome","transformer4.json").lirejson()
        self.eli5=self.eli5.flatten()
        self.block_size=128
    def preprocess_function(self,examples):
        return self.tokenizer([" ".join(x) for x in examples["answers.text"]])
    def apply_preprocess_function(self):
        self.tokenized_eli5 = self.eli5.map(
            self.preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=self.eli5["train"][0].keys(),
        )
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    def apply_group_texts_function(self):
        self.lm_dataset = self.tokenized_eli5.map(self.group_texts, batched=True, num_proc=4)
    def create_examples(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
    def train(self):
        model = AutoModelForMaskedLM.from_pretrained(self.modelname)
        training_args = TrainingArguments(
            output_dir="my_awesome_eli5_mlm_model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            num_train_epochs=3,
            weight_decay=0.01,
            push_to_hub=True,
        )
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.lm_dataset["train"],
            eval_dataset=self.lm_dataset["test"],
            data_collator=self.data_collator,
        )
        self.trainer.train()
    def evaluate(self):
        eval_results = self.trainer.evaluate()
        print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    def inference(self,text):
        mask_filler = pipeline("fill-mask", "my_awesome_eli5_mlm_model")
        return mask_filler(text, top_k=3)
ai=Transformer5()
ai.apply_preprocess_function()
ai.apply_group_texts_function()
ai.create_examples()
ai.train()
ai.evaluate()
text = "The Milky Way is a <mask> galaxy."
print(ai.inference(text))
        """
        Fichier("./","transformer5.py").ecrire(transformer5)
        Fichier("./uploads","transformer6.sh").ecrire("pip3 install transformers datasets evaluate sacrebleu")
        tr=[{'id': '90560',
         'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
           'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}]
        x_train, x_test, y_train, y_test = train_test_split(
            tr, tr, test_size=0.2, random_state=0
        )
        squad={"train":x_train, "test":x_test}
        Fichier("./uploads","transformer6.json").ecrirejson(squad)
        transformer6="""from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
from transformers import AutoModelForMaskedLM
from transformers import pipeline
from transformers import DataCollatorForSeq2Seq
import math
class Transformer6():
    def __init__(self,checkpoint="google-t5/t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.checkpoint=checkpoint
        self.source_lang = "en"
        self.target_lang = "fr"
        self.prefix = "translate English to French: "
        self.books=Fichier("./welcome","transformer6.json").lirejson()
        self.block_size=128
    def preprocess_function(self,examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs
    def apply_preprocess_function(self,examples):
        self.tokenized_books = self.books.map(self.preprocess_function, batched=True)
    def create_examples(self):
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.checkpoint)
    def evaluate(self):
        self.metric = evaluate.load("sacrebleu")
    def postprocess_text(self,preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        return preds, labels
    def compute_metrics(self,eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    def train(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        training_args = Seq2SeqTrainingArguments(
            output_dir="my_awesome_opus_books_model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=2,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_books["train"],
            eval_dataset=self.tokenized_books["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
    def inference(self,text):
        translator = pipeline("translation_"+self.source_lang+"_to_"+self.target_lang, model="my_awesome_opus_books_model")
        return translator(text)
ai = Transformer6()
ai.apply_preprocess_function()
ai.create_examples()
ai.evaluate()
ai.train()
text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
print(ai.inference(text))
            """
       Fichier("./","transformer6.py").ecrire(transformer6)
       Fichier("./uploads","transformer7.sh").ecrire("pip3 install transformers datasets evaluate rouge_score")
       billsum=[{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}]
        x_train, x_test, y_train, y_test = train_test_split(
            billsum, billsum, test_size=0.2, random_state=0
        )
        squad={"train":x_train, "test":x_test}
        Fichier("./uploads","transformer7.json").ecrirejson(squad)
        transformer7="""from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np
from transformers import AutoModelForMaskedLM
from transformers import pipeline
from transformers import DataCollatorForSeq2Seq
import math
class Transformer7():
    def __init__(self,checkpoint="google-t5/t5-small"):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.checkpoint=checkpoint
        self.prefix = "summary: "
        self.billsum=Fichier("./welcome","transformer7.json").lirejson()
        self.block_size=128
    def preprocess_function(self,examples):
        inputs = [self.prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    def apply_preprocess_function(self,examples):
        self.tokenized_billsum = self.billsum.map(self.preprocess_function, batched=True)
    def create_examples(self):
        self.data_collator = datacollatorforseq2seq(tokenizer=self.tokenizer, model=self.checkpoint)
    def train(self):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint) 
        training_args = Seq2SeqTrainingArguments(
            output_dir="my_awesome_billsum_model",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=True,
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=self.tokenized_billsum["train"],
            eval_dataset=self.tokenized_billsum["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
    def inference(self,text):
        summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")
        return summarizer(text)
ai=Transformer7()
ai.apply_preprocess_function()
ai.create_examples()
ai.train()
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
print(ai.inference(text))


        """
