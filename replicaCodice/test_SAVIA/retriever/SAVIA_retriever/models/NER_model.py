from gliner import GLiNER
import os


class NERModel():
    def __init__(self, configs):
        self.configs = configs
        self.model, self.labels = self.load_NER_model()

    def load_NER_model(self):

        NER_model_name = self.configs['models']['NER_model']
        embedding_model_path = os.path.join(self.configs['models']['models_folder'], NER_model_name.split("/")[-1])

        if NER_model_name.split("/")[-1] in os.listdir(self.configs['models']['models_folder']):
            if self.configs['verbose']:
                print("loading", NER_model_name, "NER model from:", embedding_model_path)
            NER_model = GLiNER.from_pretrained(embedding_model_path, device = "cuda")

        else:
            if self.configs['verbose']:
                print("loading", NER_model_name, "NER model from HF")
            NER_model = GLiNER.from_pretrained("DeepMount00/GLiNER_ITA_LARGE", device = "cuda")

            if self.configs['verbose']:
                print("saving embedding model in:", embedding_model_path)
            NER_model.save_pretrained(embedding_model_path)
        
        NER_labels = ["anno"]

        return NER_model, NER_labels


    def extract_entities(self, question):

        entities = self.model.predict_entities(question, self.labels)

        threshold_NER = self.configs['NER']['NER_threshold']
        entities = [x for x in entities if x['score'] > threshold_NER]


        if self.configs['verbose'] and len(entities) > 0:
            print("Entities:")
            for entity in entities:
                print(entity["text"], "=>", entity["label"])

        entities_anno = [x['text'] for x in entities if x['label'] == 'anno']

        return entities, entities_anno

    
