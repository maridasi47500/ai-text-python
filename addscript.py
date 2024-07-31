from country import Country
from urllib.request import Request, urlopen
from chercherimage import Chercherimage
from bs4 import BeautifulSoup
from fichier import Fichier
from pics import Pics
from somescaffold import Scaffold
import json

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
