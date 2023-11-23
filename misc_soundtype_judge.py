class PINYIN: 
    # In pinyin, all rhymes have a tone marker. So just detect numbers. 
    @staticmethod
    def vowel_consonant(transcription): 
        pinyin_consonants = ['b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's']
        if transcription[-1].isdigit():
            return "vowel"
        elif transcription in pinyin_consonants: 
            return "consonant"
        else: 
            return "nap"

class SAMPA: 
    @staticmethod
    def is_vowel(transcription):
        vowels = ['aa', 'aan', 'ae', 'aen', 'ah', 'ahn', 'ao', 'aon', 'aw', 'awn', 'ay', 'ayn', 'eh', 'ey', 'eyn', 'ih', 'ihn', 'iy', 'iyn', 'ow', 'own', 'oy', 'oyn', 'uh', 'uhn', 'uw', 'uwn', 'w', 'y', 'ehn', 'er', 'ern']
        
        if transcription in vowels:
            return True
        else:
            return False
    
    @staticmethod
    def is_consonant(transcription):
        consonants = ['b', 'ch', 'd', 'dh', 'dx', 'el', 'em', 'en', 'f', 'g', 'hh', 'jh', 'k', 'l', 'm', 'n', 'ng', 'nx', 'p', 'r', 's', 'sh', 't', 'th', 'tq', 'v', 'z', 'zh']

        if transcription in consonants:
            return True
        else:
            return False
    
    @staticmethod
    def vowel_consonant(transcription): 
        if SAMPA.is_vowel(transcription): 
            return "vowel"
        elif SAMPA.is_consonant(transcription): 
            return "consonant"
        else: 
            return "nap"



class ARPABET: 
    @staticmethod
    def is_vowel(transcription):
        vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AX', 'AXR', 'AY', 'EH', 'ER', 'EY', 'IH', 'IX', 'IY', 'OW', 'OY', 'UH', 'UW', 'UX']
        
        if transcription in vowels:
            return True
        else:
            return False
    
    @staticmethod
    def is_consonant(transcription):
        consonants = [
            'B', 'CH', 'D', 'DH', 'DX', 'EL', 'EM', 'EN', 'F', 'G', 'HH', 'H', 'JH', 'K', 'L', 'M', 'N',
            'NX', 'NG', 'P', 'Q', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'WH', 'Y', 'Z', 'ZH'
        ]
        
        if transcription in consonants:
            return True
        else:
            return False
    
    @staticmethod
    def vowel_consonant(transcription): 
        if ARPABET.is_vowel(transcription): 
            return "vowel"
        elif ARPABET.is_consonant(transcription): 
            return "consonant"
        else: 
            return "nap"