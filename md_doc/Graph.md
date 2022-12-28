## PPQ Graph(PPQ 计算图)

PPQ 在 Onnx IR 的基础上构建了自己的计算图结构。在这里，一个计算图的定义由 图(BaseGraph)， 算子 (Operation) 和 变量 (Variable) 三者组成。 
PPQ 的计算图对象不仅保存了你网络的结构信息与参数信息，同时也保存了图上的量化信息与量化参数，所有的量化过程都是围绕着这样的底层数据结构展开的。
本文档将向你介绍 PPQ IR 的基本定义与使用方法：

Onnx 参考: https://github.com/onnx/onnx/blob/main/docs/IR.md


### 1. Variable(变量):

该类型定义于 ppq\IR\base\opdef.py。

在 Onnx 中，Variable 描述了图中的连接关系，你可以理解它是图中的 "边"。在 Onnx 的定义中，图中的边是单向且一对多的。
这意味着 Onnx 中的边总会一个或零个输入算子，而可以多余于一个输出算子。当一个 Variable 的输入算子是 None，它表示 Onnx 中的常量，通常而言这些常量用来表示网络中的权重以及参数。

#### 成员属性:
* **is_parameter(self) -> bool:**
    
    返回当前变量是否是一个参数，如果变量是一个参数，那么通常而言它是有常数值的。
    参数变量会出现在 onnx 文件的 initializer 中。这个属性允许用户进行修改，你可以通过 variable.is_parameter = True 的方式对其进行赋值。

* **name(self) -> str:**
    
    返回当前算子的名字，你可以使用 variable.name = 'New Var Name' 的方式为变量指定新的名字。
    但值得注意的是，你不能在一个图中有两个命名相同的变量。

* **value(self) -> torch.Tensor:**
    
    返回当前 variable 的值，对于那些连接了算子两端的非参数变量而言，它们通常是无值的，仅当运行时，它们的值才会被赋予，
    而对于常数变量而言，variable.value 将直接返回变量的常量值。用户可以通过 variable.value = xxx 的方式对变量的值直接进行修改。

* **dest_ops(self) -> List[Operation]:**
    
    返回当前变量连接的所有下游节点。

    在 Onnx 与 PPQ 的定义中，任何 Variable 都有且只有一个源算子，但可以有多个目标算子，该函数将返回与当前 Variable 相连接的所有目标算子，这个列表是无序的。
    用户可以直接修改这个列表从而为 Variable 添加新的目标算子，如此修改时，用户需要同步修改目标算子的 inputs 属性。

* **source_op(self) -> Operation:**
    
    返回当前变量的源算子。

    在 Onnx 与 PPQ 的定义中，任何 Variable 都有且只有一个源算子，但可以有多个目标算子，该函数将返回与当前 Variable 的源算子。
    用户可以直接修改该属性从而替换变量的源算子，如此修改时，用户需要同步修改源算子的 outputs 属性。

* **shape(self) -> List[Union[Text, int, None]]:**
    
    返回当前变量的形状信息

    当 variable.value 不为空时，该函数直接返回 variable.value.shape，当 variable.value 为空时，该函数返回自身的 self._shape 属性。
    非参数变量的形状信息将在执行时由 ppq.executor.TorchExecutor.tracing_operation_meta 方法创建，
    若用户没有调用此函数，则图中所有非参数变量的形状信息均为空值。该属性会在导出时写入 Onnx 文件。

    用户可以通过直接赋值的方式修改非参数变量的形状信息，PPQ 接受 int, Text, None 三种类型指定的形状
    
    如 ['Batchsize', 3, 224, 224], [1, 3, 224, 224], [None, 3, 224, 224] 均是合理有效的形状信息

* **dtype(self) -> DataType:**
    
    返回当前变量的数据类型。

    当 variable.value 不为空时，该函数直接返回 variable.value.dtype，当 variable.value 为空时，该函数返回自身的 self._dtype_ 属性。
    非参数变量的数据类型信息将在执行时由 ppq.executor.TorchExecutor.tracing_operation_meta 方法创建，
    若用户没有调用此函数，则图中所有非参数变量的数据类型默认为 FP32。该属性会在导出时写入 Onnx 文件。

    用户可以通过直接赋值的方式修改非参数变量的数据类型。

* **copy(self, copy_value: bool = False):**
    
    返回当前变量的克隆，参数 copy_value 指明了是否进行深克隆。

### 2. Operation(算子):

类型定义于 ppq\IR\base\opdef.py。

Operation 是一个计算图中的节点，在 Onnx 中一个节点可以有多个输入和输出变量，任何一个算子的输入和输出都是有序的。

#### 成员属性:

* **socket(self) -> OpSocket:**
    
    返回当前算子的接线器(OpSocket)，在 PPQ 当中我们使用算子接线器描述算子应当被如何量化以及网络中的数据传递情况。
    算子接线器描述了该算子的每一个输入与输出变量是否能够被量化，描述了算子中数据从哪一个输入流动到哪一个输出。

* **inputs(self) -> List[Variable]:**

    返回当前算子的所有输入变量，若没有输入则返回空列表，该列表是有序的。
    该属性直接返回 op._inputs 对象，你可以直接修改返回的列表从而为算子添加新的输入，
    如此修改时，你需要同步修改输入变量的 dest_ops 属性。

* **outputs(self) -> List[Variable]:**
    
    返回当前算子的所有输出变量，若没有输出则返回空列表，该列表是有序的。
    该属性直接返回 op._outputs 对象，你可以直接修改返回的列表从而为算子添加新的输出，如此修改时，你需要同步修改输出变量的 source_op 属性。

* **parameters(self) -> List[Variable]:**
    
    返回当前算子的所有参数，这将返回一个视图，你并不能修改这个视图来为算子添加新的参数。

* **num_of_parameter(self) -> int:**
    
    返回当前算子的参数个数。

* **name(self) -> str:**
    
    返回当前算子的名字。用户可以使用 op.name = 'New Op Name' 的方式为算子指定新的名字，但值得注意的是，你不能在一个图中有两个命名相同的算子。

* **type(self) -> str:**
    
    返回当前算子的类型。用户可以使用 op.type = 'New Op Type' 的方式为算子指定新的类型。

* **opset(self) -> Opset:**
    
    返回当前算子的 Opset。

* **attributes(self) -> Dict[str, Any]:**
    
    返回当前算子的属性字典。你可以对其进行修改从而添加、修改、删除算子上的属性，任何添加的属性都会在导出时被写入 Onnx 文件中。

* **platform(self) -> TargetPlatform:**
    
    返回当前算子的平台。你可以使用 op.platform = xxx 来修改算子的调度情况。

* **num_of_input(self) -> int:**
    
    返回当前算子的输入个数。

* **num_of_output(self) -> int:**
    
    返回当前算子的输出个数。

* **extension_attrib(self) -> dict:**
    
    返回扩展属性集合，该属性与 attributes 类似，但写入此处的属性并不会写入到 Onnx 中。

    用户可以使用 extension_attrib 为算子添加自定义的信息，从而完成信息在不同 Optim Pass 之间的传递。

#### 成员函数：

* **copy(self) -> Operation:**

    返回一个当前算子的克隆对象

### 3. QuantableOperation(量化算子):

类型定义于 ppq\IR\quantize.py。

QuantableOperation 是 PPQ 在 Onnx Operation 基础上追加定义的内容，它表示一个被量化过的算子，
它是 Operation 的子类，这表示它具有普通算子的一切功能，于此同时我们追加地定义了一些用于量化的内容：

#### 成员属性:

* **config(self) -> OperationQuantizationConfig:**
    
    返回当前量化算子的量化信息，OperationQuantizationConfig 结构体包含了量化算子的所有输入、输出量化信息。

* **input_quant_config(self) -> List[TensorQuantizationConfig]:**
    
    以有序列表的形式返回当前算子的输入量化信息。用户可以修改量化信息中的内容，但不推荐修改列表本身。

* **output_quant_config(self) -> List[TensorQuantizationConfig]:**
    
    以有序列表的形式返回当前算子的输出量化信息。用户可以修改量化信息中的内容，但不推荐修改列表本身。

#### 成员函数：

* **baking_parameters(self, quant_func: BaseQuantFunction):**
    
    执行量化烘焙，这将使得该算子的所有参数量化静态化，这些参数的值将被量化后的值所替换，其对应 TQC 的状态将被切换为 Baked。
    在神经网络的前向传播中，我们总是需要反复计算算子参数的量化过程，但当参数已经确定并不再发生变化时，我们可以通过缓存这些参数量化结果的方式加速程序运行。

* **store_parameter_value(self):**
    
    备份算子当前的状态，该函数与 dequantize 相互呼应，使得算子能够在量化与非量化状态中相互切换

* **dequantize(self, parameter_only: bool = False, expire_device: str = 'cpu'):**
    
    解除当前算子的量化，如果在量化过程中算子的参数并未发生改变，则可以通过直接切换 TQC 状态的方式完成 dequantize 操作。
    但如果在量化过程中算子的参数被修改，则该方法不仅切换 TQC 的状态，同时从 expire_device 上取得算子参数的备份数据并替换现有的。

* **restore_quantize_state(self, expire_device: str = 'cpu'):**
    
    还原算子的量化，如果在量化过程中算子的参数并未发生改变，则可以通过直接切换 TQC 状态的方式完成 restore_quantize_state 操作。
    但如果在量化过程中算子的参数被修改，则该方法不仅切换 TQC 的状态，同时从 expire_device 上取得量化后的算子参数并替换现有的。

### 4. Graph(计算图):

类型定义于 ppq\IR\base\opdef.py。

#### 成员属性:
* **operations -> Dict[str, Operation]:**

    返回一个字典，包含当前计算图中的所有算子。如需添加或删除一个算子，用户需要使用成员函数 create_operation, remove_operation。
    用户可以使用 graph.operations.values() 方法来获得一个算子列表从而进行算子遍历

* **variables -> Dict[str, Variable]:**

    返回一个字典，包含当前计算图中的所有变量。如需添加或删除一个算子，用户需要使用成员函数 create_variable, remove_variable。
    用户可以使用 graph.variables.values() 方法来获得一个算子列表从而进行算子遍历

* **inputs -> Dict[str, Variable]:**

    返回一个字典，包含当前计算图中的所有输入变量。如需添加或删除输入变量，用户可以直接操作该字典对象，或者通过成员函数 mark_variable_as_graph_input 完成。
    理论上你可以将任何图中的变量标记为图的输入变量，但这可能导致错误，请不要将参数与中间变量标记为输入。

* **outputs -> Dict[str, Variable]:**

    返回一个字典，包含当前计算图中的所有输出变量。如需添加或删除一个算子，用户可以直接操作该字典对象，或者通过成员函数 mark_variable_as_graph_output 完成。
    你可以将任何变量标记为图的输出，但这可能导致错误，并干扰图的融合逻辑。当你把图中的中间变量标记为输出时，请格外注意该操作是否会对图融合产生影响，PPQ 与 推理框架都难以保证此时图融合的正确性。

#### 成员函数：
* **get_downstream_operations(self, operation: Operation) -> List[Operation]:**

    获取一个给定算子的所有下游算子，如算子不在图中，将抛出异常。
    如果给定的算子没有下游算子，则返回空的列表。

* **get_upstream_operations(self, operation: Operation) -> List[Operation]:**

    获取一个给定算子的所有上游算子，如算子不在图中，将抛出异常。
    如果给定的算子没有上游算子，则返回空的列表。

* **topological_sort(self) -> List[Operation]:**

    返回一个算子列表，其中的算子按照图内拓扑序先后进行排序。
    对一个有向无环图 G 进行拓扑排序，是将 G 中所有顶点排成一个线性序列，使得图中任意一对顶点u和v，若边<u,v>∈E(G)，则u在线性序列中出现在v之前。

    当无法完成上述操作时，则应任务图是由多个联通部分组成的，或图中有环。
    PPQ 会针对上述异常情况抛出错误，并指出那些算子无法完成排序。

* **insert_op_before(self, A: Operation, B: Operation, input_idx: int = 0):**

    在给定的算子 B 之前插入一个算子 A，被插入的算子必须不含任何输入输出变量
    如果算子 B 存在多个输入变量，用户需要依靠参数 input_idx 指明插入在哪一个变量上

    如果送入的算子 A 含有任何相连的输入输出变量，该函数将抛出异常

* **insert_op_after(self, A: Operation, B: Operation, output_idx: int = 0):**

    在给定的算子 B 之后插入一个算子 A，被插入的算子必须不含任何输入输出变量
    如果算子 B 存在多个输出变量，用户需要依靠参数 output_idx 指明插入在哪一个变量上

    如果送入的算子 A 含有任何相连的输入输出变量，该函数将抛出异常

* **remove_operation(self, removing_op: Operation, keep_coherence: bool = False):**

    从图中移除一个给定的算子，被移除的算子可以包含任意多个输入输出变量。
    移除这个算子时，这个算子的参数变量将一并从图中移除。

    通常而言，移除一个算子将导致图不再连续，用户可以手动连接不连通的部分，也可以使用参数 keep_coherence = True 要求 PPQ 自动完成上述操作。
    当被移除的算子只有一个非参数输入和输出变量时，用户可以指定 keep_coherence = True，PPQ 将保持算子被移除后图的连贯性。

    当给定的算子不在图中，该函数将抛出异常。

* **def remove_variable(self, removing_var: Variable):**

    从图中移除一个给定的变量，被移除的变量可以有 source_op 与 dest_ops。
    这个函数操作将一并将 removing_var 从 source_op 中的输出中移除，这个函数操作将一并将 removing_var 从 dest_ops 中的输入中移除

    当 removing_var 不在图中，将抛出异常，当 removing_var.source_op, removing_var.dest_ops 不在图中，将抛出异常。

* **create_link_with_op(self, A: Operation, B: Operation, variable: Variable = None):**

    连接给定的 A, B 算子，如果参数 variable 不为空，则使用给定的 variable 完成连接，否则将创建一个 variable 完成连接。
    如果给定了一个 variable，则给定的变量必须以算子 A 作为起点，或者没有 source_op(此时 PPQ 将修改它的 source_op = A)，否则将抛出异常。

    对于这个函数而言，算子 A 可以为空，此时这个函数将使得 variable.source_op = None，并将 variable 加入 B 的输入变量中。
    如果算子 A, B 不在图中，将抛出异常。

* **create_link_with_var(self, A: Variable, B: Variable):**

    连接给定的 A, B 两个变量，这个操作将使用变量 A 连接 A.source_op 与 B.dest_ops，变量 B 将被移除
    执行这个操作时，PPQ 将检查 B.source_op 是否为空，若不为空则抛出异常

    如果变量 A, B 不在图中，将抛出异常。

* **mark_variable_as_graph_input(self, var: Variable):**

    将一个变量标记为图的输入

* **mark_variable_as_graph_output(self, var: Variable):**

    将一个变量标记为图的输出

* **copy(self, copy_value: bool = False) -> BaseGraph:**

    返回一个图的克隆对象，参数 copy_value 决定了是仅拷贝图的结构，还是执行深拷贝