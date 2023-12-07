from dataclasses import dataclass
from typing import List

from adpickercpex.solutions._functions_helper import flatten_string_iterable


# TODO: it might be good to apply processing (diacritics strip...) to InterestDefinition
# to avoid inconsistencies (currently, it is necesarry to define all interests in this file already 'cleaned')


@dataclass(frozen=True)
class InterestDefinition: 
    single_tokens: List[str]
    bigrams: List[str]
    
    @property
    def all_tokens(self) -> List[str]:
        return self.single_tokens + self.bigrams
    
    @property
    def all_tokens_flat(self) -> List[str]:
        return flatten_string_iterable(self.all_tokens)
    
    @property
    def bigrams_flat(self) -> List[str]:
        return flatten_string_iterable(self.bigrams)
    
    
# ------------------------------------------------------------------------------------------------------------------ 
# All new interest should be defined here in the dictionary so that 
# they can be loaded by their name (dict key) obtained from a notebook widget
# ------------------------------------------------------------------------------------------------------------------
    
_NEW_INTERESTS_DICT = {
    "example": InterestDefinition(
        single_tokens=[
            "token", "example", 
        ], 
        bigrams=[
            "bigram example",
        ]
    ), 
    # temporary interest (useful for checking subset of words, etc.)
    "tmp": InterestDefinition(
        single_tokens=["accumulator","akumulator","akumulatorovy","akumulatoru","akumulatory","autonomous","baterie","bateriovy","batteries","battery","bezemisni","bezemisnich","cxperience","cybertruck","driverless","ecopo","ehybrid","electric","electricbus","electricbuses","electriccar","electriccars","electrictrucks","electrified","electrify","electrocar","electromobility","elektricky","elektrifikace","elektrifikovaneho","elektrifikovanem","elektrifikovany","elektrifikovanym","elektrifikovanymi","elektrifikovat","elektroaut","elektroautomobil","elektroautomobilky","elektroautomobilu","elektrobomilitu","elektrobomility","elektrobus","elektrobusu","elektrobusy","elektrokamionu","elektromobil","elektromobilita","elektromobilite","elektromobilitu","elektromobilni","elektromobilu","elektromobily","elektromotor","elektrovozu","elektrovuz","elon","emission-free","eqc","esprinter","esprinteru","evalia","hybrid","hybridem","hybridni","i3","i8","ioniq","musk","muska","muskovi","muskovych","muskuv","nabijitelne","nabijitelnym","rechargeable","rechargeables","recyclability","self-driving","selfdriving","solarcity","sunroq","taycan","taycanem","taycanu","tesla","tesle","teslou","teslu","tesly" ]
                               
        , 
        bigrams=[
            
        ]
    ),
}
    
    
# ---------------------------------------------------------------------------------------------------------------------------- 
    
    
def get_interest_definition(interest_name: str, empty_on_undefined: bool = False) -> InterestDefinition: 
    """ 
    Returns definition of specified interest
    
    Wrapper around the InterestDefinition object
    """
    try: 
        return _NEW_INTERESTS_DICT[interest_name]
    except KeyError as e: 
        if not empty_on_undefined:
            raise e
        return InterestDefinition(single_tokens=[], bigrams=[])
