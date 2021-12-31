"""

* Una volta che avrai Spiccato il volo deciderai

* Sguardo verso il ciel saprai Lì a casa il cuore sentirai

* Una volta che avrai Spiccato il volo deciderai

* Sguardo verso il ciel saprai Lì a casa il cuore sentirai

* Prenderà il primo volo Verso il sole il grande uccello

* Sorvolando il grande monte ceceri Riempiendo l'universo di stupore e gloria

* Una volta che avrai Spiccato il volo allora deciderai

* Sguardo verso il ciel saprai Lì a casa il cuore sentirai

* L'uomo verrà portato dalla sua creazione Come gli uccelli verso il cielo

* Riempiendo l'universo Di stupore e gloria Una volta che avrai

* Spiccato il volo allora deciderai Sguardo verso il ciel saprai

* Lì a casa il cuore sentirai  <Sogno Di Volare>

---------------------------------------------------------------

                    PPQ Core - PPQ 核心

PPL Quantization Tool(PPQ) is a nerual network quantization tool

PPQ.core contains most of PPQ internal abstractions(data structures).
    
Do not modify codes within this directory if not necessary.
"""

from .quant import *
from .data import *
from .defs import *
from .storage import *

from .config import *
from .common import *

if USING_CUDA_KERNEL:
  from .ffi import CUDA

print("""
      ____  ____  __   ____                    __              __
     / __ \/ __ \/ /  / __ \__  ______ _____  / /_____  ____  / /
    / /_/ / /_/ / /  / / / / / / / __ `/ __ \/ __/ __ \/ __ \/ / 
   / ____/ ____/ /__/ /_/ / /_/ / /_/ / / / / /_/ /_/ / /_/ / /  
  /_/   /_/   /_____\___\_\__,_/\__,_/_/ /_/\__/\____/\____/_/ 

""")