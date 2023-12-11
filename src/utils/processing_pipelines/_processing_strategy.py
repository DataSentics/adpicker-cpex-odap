"""
Simple enum definition of the processing version 

The definition had to be isolated to avoid circular imports
"""

from enum import Enum


FLAG_STEMMING = 0b01
FLAG_BIGRAM = 0b10

    
class ProcessingStrategy(Enum): 
    """
    Determines a way of processing the data into tokens & processing of interests
    """
    
    """
    Both data and interest definitions are cleaned, bigrams are not used
    """
    CLEANED_SINGLE = 0b00
    
    """
    Both data and interest definitions are stemmed, bigrams are not used
    """
    STEMMED_SINGLE = 0b01
    
    """
    Both data and interest definitions are cleaned, bigrams are used
    """
    CLEANED_BIGRAM = 0b10
    
    """
    Both data and interest definitions are stemmed, bigrams are used
    """
    STEMMED_BIGRAM = 0b11
