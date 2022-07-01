"""
                                                                                                     
                                                              ...::iirii:..                                             
                P7Ysi.  iS7u.                ..:::irri:rPBBBBBBBBBBBBBBBBBBBBBBBg57.                                    
                B   :Js:B7 .PS        :sEBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBDv                                
                B      .:     u  :5BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB2                             
                DQ            SBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBQBBBBBBBBBBBBBBBBBBBBBBBBBBRi                          
           JZ  ..i:.      .dBBBBBBBBBBBBBBBBBBBBBBBg7          .BQBBBBBBBBBBBBBBBBBBBBBBBBBBBBBD7                       
            YQ.         vBBBBBBBBBBBBBBBBBBBBBBMY.              QBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB7                    
              2D      YBBBBBBBBBBBBBBBBBBBBZ7                   BBBBBBBBBBBBBBr SBBBBBBBBBBBBBBBBBBBBM:                 
               :Bi  UBBBBBBBBBBBBBBBBBPv.                       BBBBBBBBBBBBj    .BBBBBBBBBBBBBBBBBBBBBBY               
                 ZQBBBBBBBBBBBBBBE                             DBBBBBBBBBBS        .PBBBBBBBBBBBBBBBBBBBBBu             
                ZBBBBBBBBBBBBBBBZ                             EBBBBBBBBb:             :PBBBBBBBBBBBBBBBBBBBB            
             .iBBBBBBBBBBBBBBBBB          ..:i:.            .BBBBBQKr                     7gBBBBBBBBBBBBBBBBB           
             gBBBBBBBBBBBBBBBBQ      .irri::...           IR2r.                   rir7j17.    :LBBBBQRQBBBBBBB          
             BBBBBBBBBBBBBBBB:     rur.                 .i.                              iKg7     vBBBBBBBBBBB:         
            BBBBBBBBBBBBBBP.      Xr    ..:r7rrr.                              .::LKIvrr.   5Br     :BBBBBBBBB1         
           bBBBBBBBBBBZ7.            :Yr.     :dBBP                         .JYi.      :PBE   E       BBBBBBBBr        
          jBBBBBBBBBB              :Q7    .rv:   vBb                        Qi            BB:         :BBBBBBB.        
          BBBBBBBBBBr              B.   BBBBBBBBQ                                5BBBBBQi  dB          IBBBBQB5        
         :BBBBBBBBBB                   JBBBBBBBBB:                              BBBBBBBBB:               rBBBBB        
          BBBBBBBBB1                    XBBBBBBU                                ZBBBBBBBB.                EBBBB7MQ     
          BBBBBBBBB7                                                              .LU1r.                  iBBBb  JBv   
          BBBBBBBBg                  5D7                         ..                        .              :BBB7   rB    
          qB: gBBBK           i  : .  .Y21JJL7i.                 ..B:             ULi::iirr:..            .BBu  :  B7  
        gd Ui  QBBE            .   :.                              r.              .....    :i. :.        PBR     .B.  
       7B .    iBBB                                                                             .        .B 7r    BL   
       BB .     UBBv                                                                                     BP .   rBr    
       MB      q  .                                                                                     XS    rDj:      
       .BB    .::. :                  iLv7ri:.                                      ...                ..  :YUr         
         BB7     .:U                   ...irvUXPP27:                      ..:71SqP2Is7i              Ui::rr:          
          .BBg:                                  .rLr:r7i..        .:irrrii:..                      Bg                
             7QBE7.                                     .:r77v .rri:.                             rB7                 
                :JdDE277S.                                   ..r                                rDK                   
                      . rBBi                                                                .is5r                     
                          :gBEr                                                          :7jL:                        
                             :SQMSr.                                                 .:7r:.                            
                                 :YKPbur::..                                 ...irrqBq.                                 
                                      :KgBQU1UUJr:          i7777vsv777v777ri:...    rdQJ.                              
                                    7qj:                          ..              .     7QBr                            
                                  rP:                                                      bB            

Sensetime PPL Quant Tool(PPQ) is a neural network quantization tool for high-performance deep learning inference.
It includes necessary network parsers, quant-simulator and optimization algorithms.

Generally, a quantized neural network will run 4x faster with 75% less memory cost than its float version.
However quantization is not always safe, sometime you will find there is a accuracy drop from quantization.

This tool is designed thus, for solving problems during quantization.

PPQ.core contains most of PPQ internal abstractions(data structures).

Do not modify codes within this directory if not necessary.
"""

from .common import *
from .config import PPQ_CONFIG
from .data import (DataType, OperationMeta, TensorMeta, convert_any_to_numpy,
                   convert_any_to_python_primary_type, convert_any_to_string,
                   convert_any_to_torch_tensor, convert_primary_type_to_list)
from .defs import (SingletonMeta, empty_ppq_cache, ppq_debug_function,
                   ppq_file_io, ppq_info, ppq_legacy,
                   ppq_quant_param_computing_function, ppq_warning)
from .ffi import CUDA
from .quant import (ChannelwiseTensorQuantizationConfig, NetworkFramework,
                    OperationQuantizationConfig, QuantizationPolicy,
                    QuantizationProperty, QuantizationStates, RoundingPolicy,
                    TargetPlatform, TensorQuantizationConfig)
from .storage import (Serializable, ValueState, is_file_exist,
                      open_txt_file_from_writing)
from typing import Any, Callable, List, Iterable, Set, Dict, Union, Text

print("""
      ____  ____  __   ____                    __              __
     / __ \/ __ \/ /  / __ \__  ______ _____  / /_____  ____  / /
    / /_/ / /_/ / /  / / / / / / / __ `/ __ \/ __/ __ \/ __ \/ /
   / ____/ ____/ /__/ /_/ / /_/ / /_/ / / / / /_/ /_/ / /_/ / /
  /_/   /_/   /_____\___\_\__,_/\__,_/_/ /_/\__/\____/\____/_/

""")
