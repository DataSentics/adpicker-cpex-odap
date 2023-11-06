# fmt: off
stop_words_en = [
"about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "article", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "condition", "did", "do", "does", "doing", "don", "down", "during", "each", "false", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "position", "recent", "s", "same", "she", "should", "so", "some", "such", "source", "t", "term", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "true", "type", "under", "until", "up", "update", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves", "content", "medium", "red",
]

stop_words_de = [
"a", "ab", "aber", "ach", "acht", "achte", "achten", "achter", "achtes", "ag", "alle", "allein", "allem", "allen", "aller", "allerdings", "alles", "allgemeinen", "als", "also", "am", "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes", "anderm", "andern", "anderr", "anders", "au", "auch", "auf", "aus", "ausser", "ausserdem", "b", "bald", "bei", "beide", "beiden", "beim", "beispiel", "bekannt", "bereits", "besonders", "besser", "besten", "bevor", "bin", "bis", "bisher", "bist", "c", "d", "d.h", "da", "dabei", "dadurch", "dafur", "dagegen", "daher", "dahin", "dahinter", "damals", "damit", "danach", "daneben", "dank", "dann", "daran", "darauf", "daraus", "darf", "darfst", "darin", "daruber", "darum", "darunter", "das", "dasein", "daselbst", "dass", "dasselbe", "davon", "davor", "dazu", "dazwischen", "dein", "deine", "deinem", "deinen", "deiner", "deines", "dem", "dementsprechend", "demgegenuber", "demgemass", "demselben", "demzufolge", "den", "denen", "denn", "denselben", "der", "deren", "derer", "derjenige", "derjenigen", "dermassen", "derselbe", "derselben", "des", "deshalb", "desselben", "dessen", "deswegen", "detail", "dich", "die", "diejenige", "diejenigen", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen", "dieser", "dieses", "dir", "doch", "dort", "drei", "drin", "dritte", "dritten", "dritter", "drittes", "du", "durch", "durchaus", "durfen", "durft", "durfte", "durften", "e", "eben", "ebenso", "ehrlich", "ei", "ei,", "eigen", "eigene", "eigenen", "eigener", "eigenes", "ein", "einander", "eine", "einem", "einen", "einer", "eines", "einig", "einige", "einigem", "einigen", "einiger", "einiges", "einmal", "eins", "elf", "en", "ende", "endlich", "entweder", "er", "ernst", "erst", "erste", "ersten", "erster", "erstes", "es", "etwa", "etwas", "euch", "euer", "eure", "eurem", "euren", "eurer", "eures", "f", "folgende", "fruher", "funf", "funfte", "funften", "funfter", "funftes", "fur", "g", "gab", "ganz", "ganze", "ganzen", "ganzer", "ganzes", "gar", "gedurft", "gegen", "gegenuber", "gehabt", "gehen", "geht", "gekannt", "gekonnt", "gemacht", "gemocht", "gemusst", "genug", "gerade", "gern", "gesagt", "geschweige", "gewesen", "gewollt", "geworden", "gibt", "ging", "gleich", "gott", "gross", "grosse", "grossen", "grosser", "grosses", "gut", "gute", "guter", "gutes", "h", "hab", "habe", "haben", "habt", "hast", "hat", "hatte", "hatten", "hattest", "hattet", "heisst", "her", "heute", "heuteat", "hier", "hin", "hinter", "hoch", "i", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "ihres", "im", "immer", "in", "indem", "infolgedessen", "ins", "irgend", "ist", "j", "ja", "jahr", "jahre", "jahren", "je", "jede", "jedem", "jeden", "jeder", "jedermann", "jedermanns", "jedes", "jedoch", "jemand", "jemandem", "jemanden", "jene", "jenem", "jenen", "jener", "jenes", "jetzt", "k", "kam", "kann", "kannst", "kaum", "kein", "keine", "keinem", "keinen", "keiner", "keines", "kleine", "kleinen", "kleiner", "kleines", "kommen", "kommt", "konnen", "konnt", "konnte", "konnten", "kurz", "l", "lang", "lange", "leicht", "leide", "lieber", "los", "m", "machen", "macht", "machte", "mag", "magst", "mahn", "mal", "man", "manche", "manchem", "manchen", "mancher", "manches", "mann", "mehr", "mein", "meine", "meinem", "meinen", "meiner", "meines", "mensch", "menschen", "mich", "mir", "mit", "mittel", "mochte", "mochten", "mogen", "moglich", "mogt", "morgen", "muss", "mussen", "musst", "musste", "mussten", "n", "na", "nach", "nachdem", "nahm", "naturlich", "neben", "nein", "neue", "neuen", "neun", "neunte", "neunten", "neunter", "neuntes", "nicht", "nichts", "nie", "niemand", "niemandem", "niemanden", "noch", "nun", "nur", "o", "ob", "oben", "oder", "offen", "oft", "ohne", "ordnung", "p", "q", "r", "recht", "rechte", "rechten", "rechter", "rechtes", "richtig", "rund", "s", "sa", "sache", "sagt", "sagte", "sah", "satt", "schlecht", "schluss", "schon", "sechs", "sechste", "sechsten", "sechster", "sechstes", "sehr", "sei", "seid", "seien", "sein", "seine", "seinem", "seinen", "seiner", "seines", "seit", "seitdem", "selbst", "sich", "sie", "sieben", "siebente", "siebenten", "siebenter", "siebentes", "sind", "so", "solang", "solche", "solchem", "solchen", "solcher", "solches", "soll", "sollen", "sollst", "sollt", "sollte", "sollten", "sondern", "sonst", "soweit", "sowie", "spater", "startseite", "statt", "steht", "suche", "t", "taboola", "tag", "tage", "tagen", "tat", "teil", "tel", "tritt", "trotzdem", "tun", "tut", "u", "uber", "uberhaupt", "ubrigens", "uhr", "um", "und", "und?", "uns", "unse", "unsem", "unsen", "unser", "unsere", "unserem", "unseren", "unserer", "unseres", "unses", "unter", "v", "vergangenen", "videos", "viel", "viele", "vielem", "vielen", "vielleicht", "vier", "vierte", "vierten", "vierter", "viertes", "vom", "von", "vor", "w", "wahr?", "wahrend", "wahrenddem", "wahrenddessen", "wann", "war", "ware", "waren", "warst", "wart", "warum", "was", "weg", "wegen", "weil", "weit", "weiter", "weitere", "weiteren", "weiteres", "welche", "welchem", "welchen", "welcher", "welches", "wem", "wen", "wenig", "wenige", "weniger", "weniges", "wenigstens", "wenn", "wer", "werde", "werden", "werdet", "weshalb", "wessen", "wie", "wieder", "wieso", "will", "willst", "wir", "wird", "wirklich", "wirst", "wissen", "wo", "woher", "wohin", "wohl", "wollen", "wollt", "wollte", "wollten", "worden", "wurde", "wurden", "x", "y", "z", "z.b", "zehn", "zehnte", "zehnten", "zehnter", "zehntes", "zeit", "zu", "zuerst", "zugleich", "zum", "zunachst", "zur", "zuruck", "zusammen", "zwanzig", "zwar", "zwei", "zweite", "zweiten", "zweiter", "zweites", "zwischen", "zwolf",
]

stop_words_cz = [
"aby", "ackoli", "ackoliv", "ahoj", "ale", "anebo", "ani", "aniz", "ano", "asi", "aspon", "at", "atd", "atp", "avsak", "az", "behem", "bez", "beze", "blizko", "bohuzel", "brzo", "bude", "budeme", "budes", "budete", "budou", "budu", "by", "byl", "byla", "byli", "bylo", "byly", "bys", "cau", "chce", "chceme", "chces", "chcete", "chci", "chteji", "chtit", "clanek", "clanku", "clanky", "co", "coz", "ctrnact", "ctyri", "dal", "dale", "daleko", "dalsi", "dekovat", "dekujeme", "dekuji", "den", "deset", "detail", "devatenact", "devet", "do", "dobry", "docela", "dva", "dvacet", "dvanact", "dve", "hodne", "internet", "internetu", "ja", "jak", "jako", "jakoz", "jde", "je", "jeden", "jedenact", "jedna", "jedno", "jednou", "jedou", "jeho", "jehoz", "jej", "jeji", "jejich", "jelikoz", "jemu", "jen", "jenom", "jeste", "jestli", "jestlize", "ji", "jich", "jim", "jimi", "jinak", "jsem", "jsi", "jsme", "jsou", "jste", "kam", "kde", "kdo", "kdy", "kdyby", "kdyz", "ke", "kolik", "krom", "krome", "ktera", "ktere", "kteri", "ktery", "kvuli", "ma", "maji", "mam", "mame", "mas", "mate", "me", "mezi", "mi", "mimo", "mit", "mne", "mnou", "moc", "mohl", "mohou", "moje", "moji", "mozna", "muj", "musi", "muze", "my", "na", "nad", "nade", "nam", "nami", "naproti", "nas", "nase", "nasi", "ne", "nebo", "nebyl", "nebyla", "nebyli", "nebyly", "neco", "nedela", "nedelaji", "nedelam", "nedelame", "nedelas", "nedelate", "nejak", "nejsi", "nejsou", "nekde", "nekdo", "nemaji", "nemame", "nemate", "nemel", "nemu", "neni", "nestaci", "nevadi", "nez", "nez", "nic", "nich", "nim", "nimi", "nula", "od", "ode", "okolo", "on", "ona", "oni", "ono", "ony", "osm", "osmnact", "ostatni", "pak", "patnact", "pet", "po", "pod", "podel", "podle", "pokud", "porad", "potom", "pouze", "pozde", "prave", "pred", "pres", "prese", "prestoze", "pri", "pro", "proc", "prosim", "proste", "proti", "proto", "protoze", "rika", "rovne", "se", "sedm", "sedmnact", "sekce", "sest", "sestnact", "skoro", "skrz", "smeji", "smi", "snad", "spolu", "sta", "ste", "sto", "sve", "svym", "svymi", "ta", "tady", "tak", "takhle", "taky", "tam", "tamhle", "tamhleto", "tamto", "te", "tebe", "tebou", "ted", "tedy", "tema", "ten", "ti", "tim", "timto", "tisic", "tisice", "to", "tobe", "tohle", "tomto", "tomu", "tomuto", "toto", "treba", "tri", "trinact", "trosku", "tva", "tve", "tvoje", "tvuj", "ty", "typ", "urcite", "uz", "vam", "vami", "variant", "vas", "vase", "vasi", "ve", "vecer", "vedle", "vlastne", "vsechno", "vsichni", "vubec", "vy", "vzdy", "za", "zac", "zatimco", "ze", "zobrazit", "zpet", "jake",
]

unwanted_tokens0 = [
 "articleid", "double", "inov", "referral", "windows", "net", "org", "dynamicleadbox", "nextimageindex", "filter", "posa", "campaign", "embedded", "videoplayer", "online", "php", "olol", "w3", "w", "lil", "utm", "https", "login", "vtm", "www", "region", "null", "hasdrawnboundary", "fbclid", "utmb", "limit", "index", "tendo", "recentarticles", "renne", "miniplayer", "jsp", "pageid", "zone", "asp", "info", "cpex", "default", "teh", "internetu", "template", "api", "desc", "rss", "artcomments", "user", "yzo", "mau", "seznam", "campaing", "com", "downlands", "kws", "sekce", "redirected", "brunner", "pred", "sectionid", "screenshotu", "img", "adcentrum", "kwb", "dop", "cz", "hig", "adi", "origin", "sez", "sid", "pgnum", "banner", "container", "html", "oformat", "push", "htm", "email", "szn", "noindex", "aspx", "cznull", "artcl", "mail", "fudff", "seq", "sug", "mce", "var", "mute", "inbox", "web", "req", "gid", "lng", "gofrom", "regionid", "phase", "http", "boxik", "page", "wikis", "google", "sectionprefixpostroll", "zoneidpostroll", "sectionprefixpreroll", "pipbutton", "disablesplashscreen", "linklabel", "skipoffsetpreroll", "sbrowser", "undefined", "facebook",
]

# fmt: on

unwanted_tokens = unwanted_tokens0
unwanted_tokens.extend(stop_words_en)
unwanted_tokens.extend(stop_words_cz)
unwanted_tokens.extend(stop_words_de)
