import re

from utils.helper_functions import get_dict_months
from bson.objectid import ObjectId

class RegionalLawRetriever():

    def __init__(self, db, coll="leggiRegionali"):
        self.db = db
        self.coll = coll
        self.dict_months = get_dict_months()
        self.laws = self.load_laws()

#        self.laws = [x for x in self.laws if x['_id'] == ObjectId("663f2cb8f8822e25d433e0e1")]
#        print("num laws", len(self.laws))

    def load_laws(self):

        list_laws = []
        coll = self.db[self.coll]

        for ind, doc in enumerate(coll.find()[:]):    
            out_dict = {}
            legge = doc['legge']
            out_dict['_id'] = doc['_id']
            out_dict['legge'] = legge

            _RE_ = re.compile(r'LEGGE REGIONALE\s([0-9]{1,2})\s(\w*)\s(\d{4})\s?\,?\sn.\s(\d{1,3})', re.IGNORECASE) 
            match = _RE_.search(legge)

            if match:
                giorno, mese, anno, numero = match.groups()

                out_dict['giorno'] = giorno
                out_dict['mese'] = mese
                out_dict['anno'] = anno
                out_dict['numero'] = numero
                out_dict['diciture'] = self.genera_diciture(out_dict)
                list_laws.append(out_dict)

            else:
                print("wrong", legge, doc['_id'])

        return list_laws


    def genera_diciture(self, out_dict):

        titles = []

        for legge_regionale in ['legge regionale', 'l\.r\.']:

            titles.append(legge_regionale + ' ' + out_dict['numero'] + ' del ' + out_dict['giorno'] + ' ' + out_dict['mese'] + ' ' + out_dict['anno'])
            titles.append(legge_regionale + ' del ' + out_dict['giorno'] + ' ' + out_dict['mese'] + ' ' + out_dict['anno'] + " n.\s?" + out_dict['numero'])
            titles.append(legge_regionale + ' ' + out_dict['numero'] + ' del ' + out_dict['giorno'] + ' ' + out_dict['mese'] + ' ' + out_dict['anno'])
            titles.append(legge_regionale + " " + out_dict['giorno'] + " " + out_dict['mese'] + " " + out_dict['anno'] + "\,? " + "n.\s?" + out_dict['numero'])
            titles.append(legge_regionale + " n\.? " + out_dict['numero'] +  "/" + out_dict['anno'])
            titles.append("n. " + out_dict['numero'] +  " del " + out_dict['giorno'] + " " + out_dict['mese'] + " " + out_dict['anno'])
            titles.append(legge_regionale + " n\.? " + out_dict['numero'] +  "/" + out_dict['anno'][2:] + "\s")
            titles.append(legge_regionale + " " + out_dict['numero'] +  "/" + out_dict['anno'])

            for sep in ["/", "-"]:
                mese_num = self.dict_months[out_dict['mese']] 
                data = out_dict['giorno'] + sep + mese_num + sep + out_dict['anno']

                titles.append(legge_regionale + " " + data + "\,? n\.\s?" + out_dict['numero'])
                titles.append(legge_regionale + " " + data[:-4] + data[-2:]  + "\,? n\.\s?" + out_dict['numero'])
                titles.append(legge_regionale + " " + "n\.\s?" + out_dict['numero'] + " del " + data)
                titles.append(legge_regionale + " " + "n\.\s?" + out_dict['numero'] + " del " + data[:-4] + data[-2:])

                if len(mese_num) == 1:
                    data_2 = out_dict['giorno'] + sep + "0" + mese_num + sep + out_dict['anno']
                    titles.append(legge_regionale + " " + data_2 + "\,? n\.\s?" + out_dict['numero'])
                    titles.append(legge_regionale + " " + data_2[:-4] + data_2[-2:] + "\,? n\.\s?" + out_dict['numero'])
                    titles.append(legge_regionale + " " + "n\.\s?" + out_dict['numero'] + " del " + data_2)
                    titles.append(legge_regionale + " " + "n\.\s?" + out_dict['numero'] + " del " + data_2[:-4] + data_2[-2:])

#        print(data)
#        print()
#        print(out_dict)

        titles = sorted(list(set(titles)))
#        doc['titles'] = titles

#        print(doc)
#            print(elem['titles'])

        return titles
    

    def find_all_regional_laws(self, text, find_only_in_list = False):

        if find_only_in_list:
            laws = []

            lists = self.find_lists(text)

            for ind, item in enumerate(lists[:]):
#                print(ind, " - ", item)        
                res = self.find_regional_law_in_list_item(item)
                if res:
            #            print(item)
                    print(res)
#                    print()

                    laws.append(res)
        else:
            laws = self.find_regional_laws(text)

        return laws

    def find_regional_laws(self, text):

        laws = []

        for ind_law, law in enumerate(self.laws[0:]):
            for dicitura in law['diciture']:
#                print(dicitura)
                if re.search(dicitura, text, flags=re.IGNORECASE):
                    laws.append({'_id': law['_id'], 'legge': law['legge'], 'dicitura': dicitura})
                    break
#                    if dicitura == 'l.r. 1/2024':
#                        print("here")
#                        res = re.search(dicitura, text, flags=re.IGNORECASE)
#                        print(res.start(), res.end())
#                        print(text[res.start(): res.end()])

        return laws


    def find_regional_law_in_list_item(self, text):

        res = {}

        for ind_law, law in enumerate(self.laws[0:]):
#            print(law)
            for dicitura in law['diciture']:
#                print(dicitura)
                if re.search(dicitura, text, flags=re.IGNORECASE):
#                    print("found")
#                    print(text)
#                    print(law)
                    print(ind_law)
                    res['_id'] = law['_id']
                    res['legge'] = law['legge']

                    break
            else:
                continue
            break

        return res
    
    def find_lists(self, text):
        _RE_SPLIT = re.compile(r'(?<=\n\-)(.*)(?=\n)') 
        lists = re.findall(_RE_SPLIT, text)
        lists = [x.strip() for x in lists if len(x) > 0]
        
        return lists
