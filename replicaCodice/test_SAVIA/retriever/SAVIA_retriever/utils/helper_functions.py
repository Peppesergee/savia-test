import re 

def clean_title(title):

    pattern = re.compile(r'^(.*?)\n', flags=re.DOTALL)    
    match = pattern.search(title)    
    
    if match:
        title_cleaned = match.groups()[0]
    else:
        title_cleaned = title
    
    return title_cleaned.strip()

def get_dict_months(two_digits = False):

    if two_digits:
        dict_months = {
                        "gennaio": "01",
                        "febbraio": "02",
                        "marzo": "03",
                        "aprile": "04",
                        "maggio": "05",
                        "giugno": "06",
                        "luglio": "07",
                        "agosto": "08",
                        "settembre": "09",
                        "ottobre": "10",
                        "novembre": "11",
                        "dicembre": "12"
                       }
    else:
        dict_months = {
                "gennaio": "1",
                "febbraio": "2",
                "marzo": "3",
                "aprile": "4",
                "maggio": "5",
                "giugno": "6",
                "luglio": "7",
                "agosto": "8",
                "settembre": "9",
                "ottobre": "10",
                "novembre": "11",
                "dicembre": "12"
               }
        
    return dict_months