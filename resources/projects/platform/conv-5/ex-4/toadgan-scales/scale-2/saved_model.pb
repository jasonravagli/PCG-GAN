└─
Ў§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.0.02unknown8┘ђ
ё
conv2d_25/kernelVarHandleOp*!
shared_nameconv2d_25/kernel*
dtype0*
_output_shapes
: *
shape:	@
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*
dtype0*&
_output_shapes
:	@
љ
batch_normalization_20/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*-
shared_namebatch_normalization_20/gamma
Ѕ
0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
dtype0*
_output_shapes
:@
ј
batch_normalization_20/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*,
shared_namebatch_normalization_20/beta
Є
/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
dtype0*
_output_shapes
:@
ю
"batch_normalization_20/moving_meanVarHandleOp*
shape:@*3
shared_name$"batch_normalization_20/moving_mean*
dtype0*
_output_shapes
: 
Ћ
6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
dtype0*
_output_shapes
:@
ц
&batch_normalization_20/moving_varianceVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*7
shared_name(&batch_normalization_20/moving_variance
Ю
:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
dtype0*
_output_shapes
:@
ё
conv2d_26/kernelVarHandleOp*
shape:@@*!
shared_nameconv2d_26/kernel*
dtype0*
_output_shapes
: 
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*
dtype0*&
_output_shapes
:@@
љ
batch_normalization_21/gammaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*-
shared_namebatch_normalization_21/gamma
Ѕ
0batch_normalization_21/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_21/gamma*
dtype0*
_output_shapes
:@
ј
batch_normalization_21/betaVarHandleOp*
shape:@*,
shared_namebatch_normalization_21/beta*
dtype0*
_output_shapes
: 
Є
/batch_normalization_21/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_21/beta*
dtype0*
_output_shapes
:@
ю
"batch_normalization_21/moving_meanVarHandleOp*
shape:@*3
shared_name$"batch_normalization_21/moving_mean*
dtype0*
_output_shapes
: 
Ћ
6batch_normalization_21/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_21/moving_mean*
dtype0*
_output_shapes
:@
ц
&batch_normalization_21/moving_varianceVarHandleOp*7
shared_name(&batch_normalization_21/moving_variance*
dtype0*
_output_shapes
: *
shape:@
Ю
:batch_normalization_21/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_21/moving_variance*
dtype0*
_output_shapes
:@
ё
conv2d_27/kernelVarHandleOp*!
shared_nameconv2d_27/kernel*
dtype0*
_output_shapes
: *
shape:@@
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*
dtype0*&
_output_shapes
:@@
љ
batch_normalization_22/gammaVarHandleOp*
shape:@*-
shared_namebatch_normalization_22/gamma*
dtype0*
_output_shapes
: 
Ѕ
0batch_normalization_22/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_22/gamma*
dtype0*
_output_shapes
:@
ј
batch_normalization_22/betaVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*,
shared_namebatch_normalization_22/beta
Є
/batch_normalization_22/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_22/beta*
dtype0*
_output_shapes
:@
ю
"batch_normalization_22/moving_meanVarHandleOp*
dtype0*
_output_shapes
: *
shape:@*3
shared_name$"batch_normalization_22/moving_mean
Ћ
6batch_normalization_22/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_22/moving_mean*
_output_shapes
:@*
dtype0
ц
&batch_normalization_22/moving_varianceVarHandleOp*
shape:@*7
shared_name(&batch_normalization_22/moving_variance*
dtype0*
_output_shapes
: 
Ю
:batch_normalization_22/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_22/moving_variance*
dtype0*
_output_shapes
:@
ё
conv2d_28/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:@@*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*
dtype0*&
_output_shapes
:@@
љ
batch_normalization_23/gammaVarHandleOp*-
shared_namebatch_normalization_23/gamma*
dtype0*
_output_shapes
: *
shape:@
Ѕ
0batch_normalization_23/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_23/gamma*
dtype0*
_output_shapes
:@
ј
batch_normalization_23/betaVarHandleOp*
shape:@*,
shared_namebatch_normalization_23/beta*
dtype0*
_output_shapes
: 
Є
/batch_normalization_23/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_23/beta*
dtype0*
_output_shapes
:@
ю
"batch_normalization_23/moving_meanVarHandleOp*3
shared_name$"batch_normalization_23/moving_mean*
dtype0*
_output_shapes
: *
shape:@
Ћ
6batch_normalization_23/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_23/moving_mean*
dtype0*
_output_shapes
:@
ц
&batch_normalization_23/moving_varianceVarHandleOp*
shape:@*7
shared_name(&batch_normalization_23/moving_variance*
dtype0*
_output_shapes
: 
Ю
:batch_normalization_23/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_23/moving_variance*
dtype0*
_output_shapes
:@
ё
conv2d_29/kernelVarHandleOp*!
shared_nameconv2d_29/kernel*
dtype0*
_output_shapes
: *
shape:@	
}
$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*
dtype0*&
_output_shapes
:@	
t
conv2d_29/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:	*
shared_nameconv2d_29/bias
m
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes
:	*
dtype0

NoOpNoOp
пJ
ConstConst"/device:CPU:0*ЊJ
valueЅJBєJ B I
┌
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer-19
regularization_losses
trainable_variables
	variables
	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
R
&regularization_losses
'trainable_variables
(	variables
)	keras_api
R
*regularization_losses
+trainable_variables
,	variables
-	keras_api
^

.kernel
/regularization_losses
0trainable_variables
1	variables
2	keras_api
Ќ
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8regularization_losses
9trainable_variables
:	variables
;	keras_api
R
<regularization_losses
=trainable_variables
>	variables
?	keras_api
^

@kernel
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
Ќ
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
R
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
^

Rkernel
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
Ќ
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
R
`regularization_losses
atrainable_variables
b	variables
c	keras_api
^

dkernel
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
Ќ
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
nregularization_losses
otrainable_variables
p	variables
q	keras_api
R
rregularization_losses
strainable_variables
t	variables
u	keras_api
h

vkernel
wbias
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
R
|regularization_losses
}trainable_variables
~	variables
	keras_api
V
ђregularization_losses
Ђtrainable_variables
ѓ	variables
Ѓ	keras_api
 
f
.0
41
52
@3
F4
G5
R6
X7
Y8
d9
j10
k11
v12
w13
д
.0
41
52
63
74
@5
F6
G7
H8
I9
R10
X11
Y12
Z13
[14
d15
j16
k17
l18
m19
v20
w21
ъ
regularization_losses
ёnon_trainable_variables
Ёlayers
єmetrics
trainable_variables
 Єlayer_regularization_losses
	variables
 
 
 
 
ъ
regularization_losses
ѕnon_trainable_variables
Ѕlayers
іmetrics
trainable_variables
 Іlayer_regularization_losses
	variables
 
 
 
ъ
regularization_losses
їnon_trainable_variables
Їlayers
јmetrics
trainable_variables
 Јlayer_regularization_losses
 	variables
 
 
 
ъ
"regularization_losses
љnon_trainable_variables
Љlayers
њmetrics
#trainable_variables
 Њlayer_regularization_losses
$	variables
 
 
 
ъ
&regularization_losses
ћnon_trainable_variables
Ћlayers
ќmetrics
'trainable_variables
 Ќlayer_regularization_losses
(	variables
 
 
 
ъ
*regularization_losses
ўnon_trainable_variables
Ўlayers
џmetrics
+trainable_variables
 Џlayer_regularization_losses
,	variables
\Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

.0

.0
ъ
/regularization_losses
юnon_trainable_variables
Юlayers
ъmetrics
0trainable_variables
 Ъlayer_regularization_losses
1	variables
 
ge
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
62
73
ъ
8regularization_losses
аnon_trainable_variables
Аlayers
бmetrics
9trainable_variables
 Бlayer_regularization_losses
:	variables
 
 
 
ъ
<regularization_losses
цnon_trainable_variables
Цlayers
дmetrics
=trainable_variables
 Дlayer_regularization_losses
>	variables
\Z
VARIABLE_VALUEconv2d_26/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

@0

@0
ъ
Aregularization_losses
еnon_trainable_variables
Еlayers
фmetrics
Btrainable_variables
 Фlayer_regularization_losses
C	variables
 
ge
VARIABLE_VALUEbatch_normalization_21/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_21/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_21/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_21/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
H2
I3
ъ
Jregularization_losses
гnon_trainable_variables
Гlayers
«metrics
Ktrainable_variables
 »layer_regularization_losses
L	variables
 
 
 
ъ
Nregularization_losses
░non_trainable_variables
▒layers
▓metrics
Otrainable_variables
 │layer_regularization_losses
P	variables
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

R0

R0
ъ
Sregularization_losses
┤non_trainable_variables
хlayers
Хmetrics
Ttrainable_variables
 иlayer_regularization_losses
U	variables
 
ge
VARIABLE_VALUEbatch_normalization_22/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_22/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_22/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_22/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
Z2
[3
ъ
\regularization_losses
Иnon_trainable_variables
╣layers
║metrics
]trainable_variables
 ╗layer_regularization_losses
^	variables
 
 
 
ъ
`regularization_losses
╝non_trainable_variables
йlayers
Йmetrics
atrainable_variables
 ┐layer_regularization_losses
b	variables
\Z
VARIABLE_VALUEconv2d_28/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

d0

d0
ъ
eregularization_losses
└non_trainable_variables
┴layers
┬metrics
ftrainable_variables
 ├layer_regularization_losses
g	variables
 
ge
VARIABLE_VALUEbatch_normalization_23/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_23/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE"batch_normalization_23/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE&batch_normalization_23/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
l2
m3
ъ
nregularization_losses
─non_trainable_variables
┼layers
кmetrics
otrainable_variables
 Кlayer_regularization_losses
p	variables
 
 
 
ъ
rregularization_losses
╚non_trainable_variables
╔layers
╩metrics
strainable_variables
 ╦layer_regularization_losses
t	variables
\Z
VARIABLE_VALUEconv2d_29/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_29/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

v0
w1

v0
w1
ъ
xregularization_losses
╠non_trainable_variables
═layers
╬metrics
ytrainable_variables
 ¤layer_regularization_losses
z	variables
 
 
 
ъ
|regularization_losses
лnon_trainable_variables
Лlayers
мmetrics
}trainable_variables
 Мlayer_regularization_losses
~	variables
 
 
 
А
ђregularization_losses
нnon_trainable_variables
Нlayers
оmetrics
Ђtrainable_variables
 Оlayer_regularization_losses
ѓ	variables
8
60
71
H2
I3
Z4
[5
l6
m7
ќ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

60
71
 
 
 
 
 
 
 
 
 
 
 

H0
I1
 
 
 
 
 
 
 
 
 
 
 

Z0
[1
 
 
 
 
 
 
 
 
 
 
 

l0
m1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 *
dtype0*
_output_shapes
: 
і
serving_default_input_7Placeholder*
dtype0*/
_output_shapes
:         @	*$
shape:         @	
і
serving_default_input_8Placeholder*
dtype0*/
_output_shapes
:         @	*$
shape:         @	
Р
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_7serving_default_input_8conv2d_25/kernelbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_26/kernelbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_27/kernelbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_28/kernelbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_29/kernelconv2d_29/bias*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         @	*#
Tin
2*/
_gradient_op_typePartitionedCall-10634605*/
f*R(
&__inference_signature_wrapper_10633492*
Tout
2
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
с

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_25/kernel/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp0batch_normalization_21/gamma/Read/ReadVariableOp/batch_normalization_21/beta/Read/ReadVariableOp6batch_normalization_21/moving_mean/Read/ReadVariableOp:batch_normalization_21/moving_variance/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp0batch_normalization_22/gamma/Read/ReadVariableOp/batch_normalization_22/beta/Read/ReadVariableOp6batch_normalization_22/moving_mean/Read/ReadVariableOp:batch_normalization_22/moving_variance/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp0batch_normalization_23/gamma/Read/ReadVariableOp/batch_normalization_23/beta/Read/ReadVariableOp6batch_normalization_23/moving_mean/Read/ReadVariableOp:batch_normalization_23/moving_variance/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOpConst*/
config_proto

CPU

GPU2*0,1J 8*
_output_shapes
: *#
Tin
2*/
_gradient_op_typePartitionedCall-10634649**
f%R#
!__inference__traced_save_10634648*
Tout
2
д
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_25/kernelbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_varianceconv2d_26/kernelbatch_normalization_21/gammabatch_normalization_21/beta"batch_normalization_21/moving_mean&batch_normalization_21/moving_varianceconv2d_27/kernelbatch_normalization_22/gammabatch_normalization_22/beta"batch_normalization_22/moving_mean&batch_normalization_22/moving_varianceconv2d_28/kernelbatch_normalization_23/gammabatch_normalization_23/beta"batch_normalization_23/moving_mean&batch_normalization_23/moving_varianceconv2d_29/kernelconv2d_29/bias*-
f(R&
$__inference__traced_restore_10634727*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*"
Tin
2*
_output_shapes
: */
_gradient_op_typePartitionedCall-10634728цу
л
б
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpф
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@г
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*A
_output_shapes/
-:+                           @*
T0Ѕ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: 
Љ
э
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634078

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
С
Ѕ
,__inference_conv2d_26_layer_call_fn_10632235

inputs"
statefulpartitionedcall_args_1
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*/
_gradient_op_typePartitionedCall-10632231*P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*A
_output_shapes/
-:+                           @*
Tin
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
Ў

Я
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpф
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@	г
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+                           	*
T0*
strides
*
paddingVALIDа
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	Ј
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*A
_output_shapes/
-:+                           	*
T0Б
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           	"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
С
M
1__inference_leaky_re_lu_20_layer_call_fn_10634006

inputs
identity▒
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-10632850*U
fPRN
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10632844*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         H@h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*.
_input_shapes
:         H@:& "
 
_user_specified_nameinputs
х
ѓ
9__inference_batch_normalization_20_layer_call_fn_10633996

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*/
_output_shapes
:         H@*/
_gradient_op_typePartitionedCall-10632826*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632813*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         H@*
T0"
identityIdentity:output:0*>
_input_shapes-
+:         H@::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :& "
 
_user_specified_nameinputs
в
ѓ
9__inference_batch_normalization_21_layer_call_fn_10634087

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10632338*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632337*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*A
_output_shapes/
-:+                           @*
Tin	
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
в
ѓ
9__inference_batch_normalization_21_layer_call_fn_10634096

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10632372*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632371*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*A
_output_shapes/
-:+                           @*
Tin	
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
У/
Ў
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632337

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
dtype0*
_output_shapes
: *
valueB J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs
Џ
h
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10633050

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         D@g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:         D@*
T0"
identityIdentity:output:0*.
_input_shapes
:         D@:& "
 
_user_specified_nameinputs
█
э
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10633019

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         D@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         D@"
identityIdentity:output:0*>
_input_shapes-
+:         D@::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
▓/
Ў
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633100

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
dtype0*
_output_shapes
: *
valueB ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         B@:@:@:@:@:L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *цp}?║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: *
T0█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpУ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*>
_input_shapes-
+:         B@::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: : :& "
 
_user_specified_nameinputs: : 
Џ
h
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10634177

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         F@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*.
_input_shapes
:         F@:& "
 
_user_specified_nameinputs
▓/
Ў
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633956

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
dtype0*
_output_shapes
: *
valueB J
Const_1Const*
_output_shapes
: *
valueB *
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         H@:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*>
_input_shapes-
+:         H@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
Ћ
c
G__inference_softmax_2_layer_call_and_return_conditional_losses_10633174

inputs
identityT
SoftmaxSoftmaxinputs*/
_output_shapes
:         @	*
T0a
IdentityIdentitySoftmax:softmax:0*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*.
_input_shapes
:         @	:& "
 
_user_specified_nameinputs
▄
m
C__inference_add_3_layer_call_and_return_conditional_losses_10633193

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         @	W
IdentityIdentityadd:z:0*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*I
_input_shapes8
6:         @	:         @	:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
▓/
Ў
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634132

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
dtype0*
_output_shapes
: *
valueB ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*K
_output_shapes9
7:         F@:@:@:@:@:*
T0*
U0L
Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpУ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*>
_input_shapes-
+:         F@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
х
ѓ
9__inference_batch_normalization_22_layer_call_fn_10634263

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
_output_shapes
:         D@*
Tin	
2*/
_gradient_op_typePartitionedCall-10633022*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632997*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         D@*
T0"
identityIdentity:output:0*>
_input_shapes-
+:         D@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
У/
Ў
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632499

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
dtype0*
_output_shapes
: *
valueB Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:L
Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpЙ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
Џ
h
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10633153

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         B@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*.
_input_shapes
:         B@:& "
 
_user_specified_nameinputs
▓/
Ў
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632894

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*K
_output_shapes9
7:         F@:@:@:@:@:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*>
_input_shapes-
+:         F@::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
С
M
1__inference_leaky_re_lu_23_layer_call_fn_10634534

inputs
identity▒
PartitionedCallPartitionedCallinputs*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633159*U
fPRN
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10633153*
Tout
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*.
_input_shapes
:         B@:& "
 
_user_specified_nameinputs
х
ѓ
9__inference_batch_normalization_23_layer_call_fn_10634439

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         B@*
Tin	
2*/
_gradient_op_typePartitionedCall-10633125*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633100*
Tout
2і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*>
_input_shapes-
+:         B@::::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: : 
Џ
h
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10632844

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         H@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*.
_input_shapes
:         H@:& "
 
_user_specified_nameinputs
х
ѓ
9__inference_batch_normalization_22_layer_call_fn_10634272

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10633019*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin	
2*/
_gradient_op_typePartitionedCall-10633032і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         D@"
identityIdentity:output:0*>
_input_shapes-
+:         D@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
С
Ѕ
,__inference_conv2d_28_layer_call_fn_10632559

inputs"
statefulpartitionedcall_args_1
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*A
_output_shapes/
-:+                           @*/
_gradient_op_typePartitionedCall-10632555ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs
У/
Ў
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634484

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *цp}?║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
ЄЕ
┬
I__inference_generator_2_layer_call_and_return_conditional_losses_10633656
inputs_0
inputs_1,
(conv2d_25_conv2d_readvariableop_resource2
.batch_normalization_20_readvariableop_resource4
0batch_normalization_20_readvariableop_1_resourceG
Cbatch_normalization_20_assignmovingavg_read_readvariableop_resourceI
Ebatch_normalization_20_assignmovingavg_1_read_readvariableop_resource,
(conv2d_26_conv2d_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceG
Cbatch_normalization_21_assignmovingavg_read_readvariableop_resourceI
Ebatch_normalization_21_assignmovingavg_1_read_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource2
.batch_normalization_22_readvariableop_resource4
0batch_normalization_22_readvariableop_1_resourceG
Cbatch_normalization_22_assignmovingavg_read_readvariableop_resourceI
Ebatch_normalization_22_assignmovingavg_1_read_readvariableop_resource,
(conv2d_28_conv2d_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceG
Cbatch_normalization_23_assignmovingavg_read_readvariableop_resourceI
Ebatch_normalization_23_assignmovingavg_1_read_readvariableop_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource
identityѕб:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpб:batch_normalization_20/AssignMovingAvg/Read/ReadVariableOpб5batch_normalization_20/AssignMovingAvg/ReadVariableOpб<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpб<batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOpб7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpб%batch_normalization_20/ReadVariableOpб'batch_normalization_20/ReadVariableOp_1б:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpб:batch_normalization_21/AssignMovingAvg/Read/ReadVariableOpб5batch_normalization_21/AssignMovingAvg/ReadVariableOpб<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpб<batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOpб7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpб%batch_normalization_21/ReadVariableOpб'batch_normalization_21/ReadVariableOp_1б:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpб:batch_normalization_22/AssignMovingAvg/Read/ReadVariableOpб5batch_normalization_22/AssignMovingAvg/ReadVariableOpб<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOpб<batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOpб7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpб%batch_normalization_22/ReadVariableOpб'batch_normalization_22/ReadVariableOp_1б:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpб:batch_normalization_23/AssignMovingAvg/Read/ReadVariableOpб5batch_normalization_23/AssignMovingAvg/ReadVariableOpб<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpб<batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOpб7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpб%batch_normalization_23/ReadVariableOpб'batch_normalization_23/ReadVariableOp_1бconv2d_25/Conv2D/ReadVariableOpбconv2d_26/Conv2D/ReadVariableOpбconv2d_27/Conv2D/ReadVariableOpбconv2d_28/Conv2D/ReadVariableOpб conv2d_29/BiasAdd/ReadVariableOpбconv2d_29/Conv2D/ReadVariableOpј
zero_padding2d_6/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:Є
zero_padding2d_6/PadPadinputs_0&zero_padding2d_6/Pad/paddings:output:0*/
_output_shapes
:         J	*
T0ј
zero_padding2d_7/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:Є
zero_padding2d_7/PadPadinputs_1&zero_padding2d_7/Pad/paddings:output:0*/
_output_shapes
:         J	*
T0і
	add_2/addAddV2zero_padding2d_6/Pad:output:0zero_padding2d_7/Pad:output:0*/
_output_shapes
:         J	*
T0Й
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:	@х
conv2d_25/Conv2DConv2Dadd_2/add:z:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         H@*
T0*
strides
e
#batch_normalization_20/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: e
#batch_normalization_20/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: Б
!batch_normalization_20/LogicalAnd
LogicalAnd,batch_normalization_20/LogicalAnd/x:output:0,batch_normalization_20/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@_
batch_normalization_20/ConstConst*
valueB *
dtype0*
_output_shapes
: a
batch_normalization_20/Const_1Const*
valueB *
dtype0*
_output_shapes
: э
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_25/Conv2D:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0%batch_normalization_20/Const:output:0'batch_normalization_20/Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         H@:@:@:@:@:c
batch_normalization_20/Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: У
:batch_normalization_20/AssignMovingAvg/Read/ReadVariableOpReadVariableOpCbatch_normalization_20_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ц
/batch_normalization_20/AssignMovingAvg/IdentityIdentityBbatch_normalization_20/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0Ь
,batch_normalization_20/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*M
_classC
A?loc:@batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp*
dtype0»
*batch_normalization_20/AssignMovingAvg/subSub5batch_normalization_20/AssignMovingAvg/sub/x:output:0'batch_normalization_20/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*M
_classC
A?loc:@batch_normalization_20/AssignMovingAvg/Read/ReadVariableOpа
5batch_normalization_20/AssignMovingAvg/ReadVariableOpReadVariableOpCbatch_normalization_20_assignmovingavg_read_readvariableop_resource;^batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╩
,batch_normalization_20/AssignMovingAvg/sub_1Sub=batch_normalization_20/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_20/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@х
*batch_normalization_20/AssignMovingAvg/mulMul0batch_normalization_20/AssignMovingAvg/sub_1:z:0.batch_normalization_20/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@ъ
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpCbatch_normalization_20_assignmovingavg_read_readvariableop_resource.batch_normalization_20/AssignMovingAvg/mul:z:06^batch_normalization_20/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*M
_classC
A?loc:@batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 В
<batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpEbatch_normalization_20_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@е
1batch_normalization_20/AssignMovingAvg_1/IdentityIdentityDbatch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ы
.batch_normalization_20/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*O
_classE
CAloc:@batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: х
,batch_normalization_20/AssignMovingAvg_1/subSub7batch_normalization_20/AssignMovingAvg_1/sub/x:output:0'batch_normalization_20/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*O
_classE
CAloc:@batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOpд
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOpReadVariableOpEbatch_normalization_20_assignmovingavg_1_read_readvariableop_resource=^batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@н
.batch_normalization_20/AssignMovingAvg_1/sub_1Sub?batch_normalization_20/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_20/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*O
_classE
CAloc:@batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOpй
,batch_normalization_20/AssignMovingAvg_1/mulMul2batch_normalization_20/AssignMovingAvg_1/sub_1:z:00batch_normalization_20/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*O
_classE
CAloc:@batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@е
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpEbatch_normalization_20_assignmovingavg_1_read_readvariableop_resource0batch_normalization_20/AssignMovingAvg_1/mul:z:08^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*O
_classE
CAloc:@batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Ѓ
leaky_re_lu_20/LeakyRelu	LeakyRelu+batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:         H@Й
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@╬
conv2d_26/Conv2DConv2D&leaky_re_lu_20/LeakyRelu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         F@*
T0*
strides
e
#batch_normalization_21/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
e
#batch_normalization_21/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: Б
!batch_normalization_21/LogicalAnd
LogicalAnd,batch_normalization_21/LogicalAnd/x:output:0,batch_normalization_21/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@_
batch_normalization_21/ConstConst*
dtype0*
_output_shapes
: *
valueB a
batch_normalization_21/Const_1Const*
valueB *
dtype0*
_output_shapes
: э
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_26/Conv2D:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0%batch_normalization_21/Const:output:0'batch_normalization_21/Const_1:output:0*
epsilon%oЃ:*K
_output_shapes9
7:         F@:@:@:@:@:*
T0*
U0c
batch_normalization_21/Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0У
:batch_normalization_21/AssignMovingAvg/Read/ReadVariableOpReadVariableOpCbatch_normalization_21_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ц
/batch_normalization_21/AssignMovingAvg/IdentityIdentityBbatch_normalization_21/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0Ь
,batch_normalization_21/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*M
_classC
A?loc:@batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: »
*batch_normalization_21/AssignMovingAvg/subSub5batch_normalization_21/AssignMovingAvg/sub/x:output:0'batch_normalization_21/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: а
5batch_normalization_21/AssignMovingAvg/ReadVariableOpReadVariableOpCbatch_normalization_21_assignmovingavg_read_readvariableop_resource;^batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╩
,batch_normalization_21/AssignMovingAvg/sub_1Sub=batch_normalization_21/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_21/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@х
*batch_normalization_21/AssignMovingAvg/mulMul0batch_normalization_21/AssignMovingAvg/sub_1:z:0.batch_normalization_21/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@ъ
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpCbatch_normalization_21_assignmovingavg_read_readvariableop_resource.batch_normalization_21/AssignMovingAvg/mul:z:06^batch_normalization_21/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*M
_classC
A?loc:@batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 В
<batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpEbatch_normalization_21_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@е
1batch_normalization_21/AssignMovingAvg_1/IdentityIdentityDbatch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ы
.batch_normalization_21/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*O
_classE
CAloc:@batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOpх
,batch_normalization_21/AssignMovingAvg_1/subSub7batch_normalization_21/AssignMovingAvg_1/sub/x:output:0'batch_normalization_21/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*O
_classE
CAloc:@batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: д
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOpReadVariableOpEbatch_normalization_21_assignmovingavg_1_read_readvariableop_resource=^batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@н
.batch_normalization_21/AssignMovingAvg_1/sub_1Sub?batch_normalization_21/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_21/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*O
_classE
CAloc:@batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOpй
,batch_normalization_21/AssignMovingAvg_1/mulMul2batch_normalization_21/AssignMovingAvg_1/sub_1:z:00batch_normalization_21/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*O
_classE
CAloc:@batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOpе
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpEbatch_normalization_21_assignmovingavg_1_read_readvariableop_resource0batch_normalization_21/AssignMovingAvg_1/mul:z:08^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*O
_classE
CAloc:@batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Ѓ
leaky_re_lu_21/LeakyRelu	LeakyRelu+batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:         F@Й
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0╬
conv2d_27/Conv2DConv2D&leaky_re_lu_21/LeakyRelu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         D@e
#batch_normalization_22/LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: e
#batch_normalization_22/LogicalAnd/yConst*
_output_shapes
: *
value	B
 Z*
dtype0
Б
!batch_normalization_22/LogicalAnd
LogicalAnd,batch_normalization_22/LogicalAnd/x:output:0,batch_normalization_22/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@_
batch_normalization_22/ConstConst*
valueB *
dtype0*
_output_shapes
: a
batch_normalization_22/Const_1Const*
valueB *
dtype0*
_output_shapes
: э
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_27/Conv2D:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0%batch_normalization_22/Const:output:0'batch_normalization_22/Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         D@:@:@:@:@:c
batch_normalization_22/Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: У
:batch_normalization_22/AssignMovingAvg/Read/ReadVariableOpReadVariableOpCbatch_normalization_22_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ц
/batch_normalization_22/AssignMovingAvg/IdentityIdentityBbatch_normalization_22/AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ь
,batch_normalization_22/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*M
_classC
A?loc:@batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp*
dtype0»
*batch_normalization_22/AssignMovingAvg/subSub5batch_normalization_22/AssignMovingAvg/sub/x:output:0'batch_normalization_22/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: а
5batch_normalization_22/AssignMovingAvg/ReadVariableOpReadVariableOpCbatch_normalization_22_assignmovingavg_read_readvariableop_resource;^batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╩
,batch_normalization_22/AssignMovingAvg/sub_1Sub=batch_normalization_22/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_22/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@х
*batch_normalization_22/AssignMovingAvg/mulMul0batch_normalization_22/AssignMovingAvg/sub_1:z:0.batch_normalization_22/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*M
_classC
A?loc:@batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0ъ
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpCbatch_normalization_22_assignmovingavg_read_readvariableop_resource.batch_normalization_22/AssignMovingAvg/mul:z:06^batch_normalization_22/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*M
_classC
A?loc:@batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 В
<batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpEbatch_normalization_22_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@е
1batch_normalization_22/AssignMovingAvg_1/IdentityIdentityDbatch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ы
.batch_normalization_22/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*O
_classE
CAloc:@batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: х
,batch_normalization_22/AssignMovingAvg_1/subSub7batch_normalization_22/AssignMovingAvg_1/sub/x:output:0'batch_normalization_22/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*O
_classE
CAloc:@batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: д
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOpReadVariableOpEbatch_normalization_22_assignmovingavg_1_read_readvariableop_resource=^batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@н
.batch_normalization_22/AssignMovingAvg_1/sub_1Sub?batch_normalization_22/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_22/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*O
_classE
CAloc:@batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOpй
,batch_normalization_22/AssignMovingAvg_1/mulMul2batch_normalization_22/AssignMovingAvg_1/sub_1:z:00batch_normalization_22/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*O
_classE
CAloc:@batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@е
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpEbatch_normalization_22_assignmovingavg_1_read_readvariableop_resource0batch_normalization_22/AssignMovingAvg_1/mul:z:08^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *O
_classE
CAloc:@batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOpЃ
leaky_re_lu_22/LeakyRelu	LeakyRelu+batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:         D@Й
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@╬
conv2d_28/Conv2DConv2D&leaky_re_lu_22/LeakyRelu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         B@*
T0*
strides
e
#batch_normalization_23/LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
e
#batch_normalization_23/LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 ZБ
!batch_normalization_23/LogicalAnd
LogicalAnd,batch_normalization_23/LogicalAnd/x:output:0,batch_normalization_23/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@_
batch_normalization_23/ConstConst*
valueB *
dtype0*
_output_shapes
: a
batch_normalization_23/Const_1Const*
valueB *
dtype0*
_output_shapes
: э
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_28/Conv2D:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0%batch_normalization_23/Const:output:0'batch_normalization_23/Const_1:output:0*
epsilon%oЃ:*K
_output_shapes9
7:         B@:@:@:@:@:*
T0*
U0c
batch_normalization_23/Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: У
:batch_normalization_23/AssignMovingAvg/Read/ReadVariableOpReadVariableOpCbatch_normalization_23_assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ц
/batch_normalization_23/AssignMovingAvg/IdentityIdentityBbatch_normalization_23/AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0Ь
,batch_normalization_23/AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*M
_classC
A?loc:@batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: »
*batch_normalization_23/AssignMovingAvg/subSub5batch_normalization_23/AssignMovingAvg/sub/x:output:0'batch_normalization_23/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*M
_classC
A?loc:@batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: а
5batch_normalization_23/AssignMovingAvg/ReadVariableOpReadVariableOpCbatch_normalization_23_assignmovingavg_read_readvariableop_resource;^batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╩
,batch_normalization_23/AssignMovingAvg/sub_1Sub=batch_normalization_23/AssignMovingAvg/ReadVariableOp:value:04batch_normalization_23/FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*M
_classC
A?loc:@batch_normalization_23/AssignMovingAvg/Read/ReadVariableOpх
*batch_normalization_23/AssignMovingAvg/mulMul0batch_normalization_23/AssignMovingAvg/sub_1:z:0.batch_normalization_23/AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*M
_classC
A?loc:@batch_normalization_23/AssignMovingAvg/Read/ReadVariableOpъ
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpCbatch_normalization_23_assignmovingavg_read_readvariableop_resource.batch_normalization_23/AssignMovingAvg/mul:z:06^batch_normalization_23/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*M
_classC
A?loc:@batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 В
<batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOpReadVariableOpEbatch_normalization_23_assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@е
1batch_normalization_23/AssignMovingAvg_1/IdentityIdentityDbatch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@Ы
.batch_normalization_23/AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*O
_classE
CAloc:@batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: х
,batch_normalization_23/AssignMovingAvg_1/subSub7batch_normalization_23/AssignMovingAvg_1/sub/x:output:0'batch_normalization_23/Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*O
_classE
CAloc:@batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: д
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOpReadVariableOpEbatch_normalization_23_assignmovingavg_1_read_readvariableop_resource=^batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@н
.batch_normalization_23/AssignMovingAvg_1/sub_1Sub?batch_normalization_23/AssignMovingAvg_1/ReadVariableOp:value:08batch_normalization_23/FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*O
_classE
CAloc:@batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0й
,batch_normalization_23/AssignMovingAvg_1/mulMul2batch_normalization_23/AssignMovingAvg_1/sub_1:z:00batch_normalization_23/AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*O
_classE
CAloc:@batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOpе
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpEbatch_normalization_23_assignmovingavg_1_read_readvariableop_resource0batch_normalization_23/AssignMovingAvg_1/mul:z:08^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *O
_classE
CAloc:@batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOpЃ
leaky_re_lu_23/LeakyRelu	LeakyRelu+batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:         B@Й
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@	╬
conv2d_29/Conv2DConv2D&leaky_re_lu_23/LeakyRelu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         @	┤
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	Џ
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @	r
softmax_2/SoftmaxSoftmaxconv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:         @	s
	add_3/addAddV2softmax_2/Softmax:softmax:0inputs_1*/
_output_shapes
:         @	*
T0џ
IdentityIdentityadd_3/add:z:0;^batch_normalization_20/AssignMovingAvg/AssignSubVariableOp;^batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp6^batch_normalization_20/AssignMovingAvg/ReadVariableOp=^batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp=^batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp8^batch_normalization_20/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1;^batch_normalization_21/AssignMovingAvg/AssignSubVariableOp;^batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp6^batch_normalization_21/AssignMovingAvg/ReadVariableOp=^batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp=^batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp8^batch_normalization_21/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_1;^batch_normalization_22/AssignMovingAvg/AssignSubVariableOp;^batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp6^batch_normalization_22/AssignMovingAvg/ReadVariableOp=^batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp=^batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp8^batch_normalization_22/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_1;^batch_normalization_23/AssignMovingAvg/AssignSubVariableOp;^batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp6^batch_normalization_23/AssignMovingAvg/ReadVariableOp=^batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp=^batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp8^batch_normalization_23/AssignMovingAvg_1/ReadVariableOp&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1 ^conv2d_25/Conv2D/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_12n
5batch_normalization_20/AssignMovingAvg/ReadVariableOp5batch_normalization_20/AssignMovingAvg/ReadVariableOp2x
:batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp:batch_normalization_23/AssignMovingAvg/Read/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2x
:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp:batch_normalization_20/AssignMovingAvg/AssignSubVariableOp2|
<batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp<batch_normalization_20/AssignMovingAvg_1/Read/ReadVariableOp2r
7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp7batch_normalization_20/AssignMovingAvg_1/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12r
7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp7batch_normalization_21/AssignMovingAvg_1/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2|
<batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp<batch_normalization_21/AssignMovingAvg_1/Read/ReadVariableOp2n
5batch_normalization_21/AssignMovingAvg/ReadVariableOp5batch_normalization_21/AssignMovingAvg/ReadVariableOp2r
7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp7batch_normalization_22/AssignMovingAvg_1/ReadVariableOp2x
:batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp:batch_normalization_22/AssignMovingAvg/Read/ReadVariableOp2r
7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp7batch_normalization_23/AssignMovingAvg_1/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2|
<batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp<batch_normalization_22/AssignMovingAvg_1/Read/ReadVariableOp2x
:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp:batch_normalization_23/AssignMovingAvg/AssignSubVariableOp2N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2|
<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_20/AssignMovingAvg_1/AssignSubVariableOp2n
5batch_normalization_22/AssignMovingAvg/ReadVariableOp5batch_normalization_22/AssignMovingAvg/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp<batch_normalization_23/AssignMovingAvg_1/Read/ReadVariableOp2|
<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_21/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp:batch_normalization_21/AssignMovingAvg/Read/ReadVariableOp2N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2x
:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp:batch_normalization_22/AssignMovingAvg/AssignSubVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2n
5batch_normalization_23/AssignMovingAvg/ReadVariableOp5batch_normalization_23/AssignMovingAvg/ReadVariableOp2|
<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_22/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp:batch_normalization_20/AssignMovingAvg/Read/ReadVariableOp2N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2|
<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp<batch_normalization_23/AssignMovingAvg_1/AssignSubVariableOp2x
:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp:batch_normalization_21/AssignMovingAvg/AssignSubVariableOp:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : : : : : : : : : : : : : 
Љ
э
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10632695

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1: :& "
 
_user_specified_nameinputs: : : 
аЎ
▓
#__inference__wrapped_model_10632017
input_7
input_88
4generator_2_conv2d_25_conv2d_readvariableop_resource>
:generator_2_batch_normalization_20_readvariableop_resource@
<generator_2_batch_normalization_20_readvariableop_1_resourceO
Kgenerator_2_batch_normalization_20_fusedbatchnormv3_readvariableop_resourceQ
Mgenerator_2_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource8
4generator_2_conv2d_26_conv2d_readvariableop_resource>
:generator_2_batch_normalization_21_readvariableop_resource@
<generator_2_batch_normalization_21_readvariableop_1_resourceO
Kgenerator_2_batch_normalization_21_fusedbatchnormv3_readvariableop_resourceQ
Mgenerator_2_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource8
4generator_2_conv2d_27_conv2d_readvariableop_resource>
:generator_2_batch_normalization_22_readvariableop_resource@
<generator_2_batch_normalization_22_readvariableop_1_resourceO
Kgenerator_2_batch_normalization_22_fusedbatchnormv3_readvariableop_resourceQ
Mgenerator_2_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource8
4generator_2_conv2d_28_conv2d_readvariableop_resource>
:generator_2_batch_normalization_23_readvariableop_resource@
<generator_2_batch_normalization_23_readvariableop_1_resourceO
Kgenerator_2_batch_normalization_23_fusedbatchnormv3_readvariableop_resourceQ
Mgenerator_2_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource8
4generator_2_conv2d_29_conv2d_readvariableop_resource9
5generator_2_conv2d_29_biasadd_readvariableop_resource
identityѕбBgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpбDgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1б1generator_2/batch_normalization_20/ReadVariableOpб3generator_2/batch_normalization_20/ReadVariableOp_1бBgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpбDgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1б1generator_2/batch_normalization_21/ReadVariableOpб3generator_2/batch_normalization_21/ReadVariableOp_1бBgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpбDgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1б1generator_2/batch_normalization_22/ReadVariableOpб3generator_2/batch_normalization_22/ReadVariableOp_1бBgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpбDgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1б1generator_2/batch_normalization_23/ReadVariableOpб3generator_2/batch_normalization_23/ReadVariableOp_1б+generator_2/conv2d_25/Conv2D/ReadVariableOpб+generator_2/conv2d_26/Conv2D/ReadVariableOpб+generator_2/conv2d_27/Conv2D/ReadVariableOpб+generator_2/conv2d_28/Conv2D/ReadVariableOpб,generator_2/conv2d_29/BiasAdd/ReadVariableOpб+generator_2/conv2d_29/Conv2D/ReadVariableOpџ
)generator_2/zero_padding2d_6/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:ъ
 generator_2/zero_padding2d_6/PadPadinput_72generator_2/zero_padding2d_6/Pad/paddings:output:0*
T0*/
_output_shapes
:         J	џ
)generator_2/zero_padding2d_7/Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0ъ
 generator_2/zero_padding2d_7/PadPadinput_82generator_2/zero_padding2d_7/Pad/paddings:output:0*
T0*/
_output_shapes
:         J	«
generator_2/add_2/addAddV2)generator_2/zero_padding2d_6/Pad:output:0)generator_2/zero_padding2d_7/Pad:output:0*/
_output_shapes
:         J	*
T0о
+generator_2/conv2d_25/Conv2D/ReadVariableOpReadVariableOp4generator_2_conv2d_25_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:	@┘
generator_2/conv2d_25/Conv2DConv2Dgenerator_2/add_2/add:z:03generator_2/conv2d_25/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         H@*
T0*
strides
*
paddingVALIDq
/generator_2/batch_normalization_20/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/generator_2/batch_normalization_20/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: К
-generator_2/batch_normalization_20/LogicalAnd
LogicalAnd8generator_2/batch_normalization_20/LogicalAnd/x:output:08generator_2/batch_normalization_20/LogicalAnd/y:output:0*
_output_shapes
: о
1generator_2/batch_normalization_20/ReadVariableOpReadVariableOp:generator_2_batch_normalization_20_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0┌
3generator_2/batch_normalization_20/ReadVariableOp_1ReadVariableOp<generator_2_batch_normalization_20_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
Bgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpKgenerator_2_batch_normalization_20_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ч
Dgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMgenerator_2_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ё
3generator_2/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3%generator_2/conv2d_25/Conv2D:output:09generator_2/batch_normalization_20/ReadVariableOp:value:0;generator_2/batch_normalization_20/ReadVariableOp_1:value:0Jgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Lgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         H@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:m
(generator_2/batch_normalization_20/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *цp}?Џ
$generator_2/leaky_re_lu_20/LeakyRelu	LeakyRelu7generator_2/batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:         H@о
+generator_2/conv2d_26/Conv2D/ReadVariableOpReadVariableOp4generator_2_conv2d_26_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0Ы
generator_2/conv2d_26/Conv2DConv2D2generator_2/leaky_re_lu_20/LeakyRelu:activations:03generator_2/conv2d_26/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         F@*
T0*
strides
q
/generator_2/batch_normalization_21/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/generator_2/batch_normalization_21/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: К
-generator_2/batch_normalization_21/LogicalAnd
LogicalAnd8generator_2/batch_normalization_21/LogicalAnd/x:output:08generator_2/batch_normalization_21/LogicalAnd/y:output:0*
_output_shapes
: о
1generator_2/batch_normalization_21/ReadVariableOpReadVariableOp:generator_2_batch_normalization_21_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┌
3generator_2/batch_normalization_21/ReadVariableOp_1ReadVariableOp<generator_2_batch_normalization_21_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
Bgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOpKgenerator_2_batch_normalization_21_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ч
Dgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMgenerator_2_batch_normalization_21_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ё
3generator_2/batch_normalization_21/FusedBatchNormV3FusedBatchNormV3%generator_2/conv2d_26/Conv2D:output:09generator_2/batch_normalization_21/ReadVariableOp:value:0;generator_2/batch_normalization_21/ReadVariableOp_1:value:0Jgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0Lgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         F@:@:@:@:@:m
(generator_2/batch_normalization_21/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Џ
$generator_2/leaky_re_lu_21/LeakyRelu	LeakyRelu7generator_2/batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:         F@о
+generator_2/conv2d_27/Conv2D/ReadVariableOpReadVariableOp4generator_2_conv2d_27_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ы
generator_2/conv2d_27/Conv2DConv2D2generator_2/leaky_re_lu_21/LeakyRelu:activations:03generator_2/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         D@q
/generator_2/batch_normalization_22/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/generator_2/batch_normalization_22/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: К
-generator_2/batch_normalization_22/LogicalAnd
LogicalAnd8generator_2/batch_normalization_22/LogicalAnd/x:output:08generator_2/batch_normalization_22/LogicalAnd/y:output:0*
_output_shapes
: о
1generator_2/batch_normalization_22/ReadVariableOpReadVariableOp:generator_2_batch_normalization_22_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┌
3generator_2/batch_normalization_22/ReadVariableOp_1ReadVariableOp<generator_2_batch_normalization_22_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
Bgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOpKgenerator_2_batch_normalization_22_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ч
Dgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMgenerator_2_batch_normalization_22_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ё
3generator_2/batch_normalization_22/FusedBatchNormV3FusedBatchNormV3%generator_2/conv2d_27/Conv2D:output:09generator_2/batch_normalization_22/ReadVariableOp:value:0;generator_2/batch_normalization_22/ReadVariableOp_1:value:0Jgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0Lgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         D@:@:@:@:@:m
(generator_2/batch_normalization_22/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Џ
$generator_2/leaky_re_lu_22/LeakyRelu	LeakyRelu7generator_2/batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:         D@о
+generator_2/conv2d_28/Conv2D/ReadVariableOpReadVariableOp4generator_2_conv2d_28_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@Ы
generator_2/conv2d_28/Conv2DConv2D2generator_2/leaky_re_lu_22/LeakyRelu:activations:03generator_2/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         B@q
/generator_2/batch_normalization_23/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: q
/generator_2/batch_normalization_23/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: К
-generator_2/batch_normalization_23/LogicalAnd
LogicalAnd8generator_2/batch_normalization_23/LogicalAnd/x:output:08generator_2/batch_normalization_23/LogicalAnd/y:output:0*
_output_shapes
: о
1generator_2/batch_normalization_23/ReadVariableOpReadVariableOp:generator_2_batch_normalization_23_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┌
3generator_2/batch_normalization_23/ReadVariableOp_1ReadVariableOp<generator_2_batch_normalization_23_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Э
Bgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOpKgenerator_2_batch_normalization_23_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ч
Dgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMgenerator_2_batch_normalization_23_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ё
3generator_2/batch_normalization_23/FusedBatchNormV3FusedBatchNormV3%generator_2/conv2d_28/Conv2D:output:09generator_2/batch_normalization_23/ReadVariableOp:value:0;generator_2/batch_normalization_23/ReadVariableOp_1:value:0Jgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0Lgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         B@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:m
(generator_2/batch_normalization_23/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Џ
$generator_2/leaky_re_lu_23/LeakyRelu	LeakyRelu7generator_2/batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:         B@о
+generator_2/conv2d_29/Conv2D/ReadVariableOpReadVariableOp4generator_2_conv2d_29_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@	Ы
generator_2/conv2d_29/Conv2DConv2D2generator_2/leaky_re_lu_23/LeakyRelu:activations:03generator_2/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*
strides
*
paddingVALID*/
_output_shapes
:         @	╠
,generator_2/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp5generator_2_conv2d_29_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	┐
generator_2/conv2d_29/BiasAddBiasAdd%generator_2/conv2d_29/Conv2D:output:04generator_2/conv2d_29/BiasAdd/ReadVariableOp:value:0*/
_output_shapes
:         @	*
T0і
generator_2/softmax_2/SoftmaxSoftmax&generator_2/conv2d_29/BiasAdd:output:0*
T0*/
_output_shapes
:         @	і
generator_2/add_3/addAddV2'generator_2/softmax_2/Softmax:softmax:0input_8*
T0*/
_output_shapes
:         @	о

IdentityIdentitygenerator_2/add_3/add:z:0C^generator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpE^generator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12^generator_2/batch_normalization_20/ReadVariableOp4^generator_2/batch_normalization_20/ReadVariableOp_1C^generator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpE^generator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12^generator_2/batch_normalization_21/ReadVariableOp4^generator_2/batch_normalization_21/ReadVariableOp_1C^generator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpE^generator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12^generator_2/batch_normalization_22/ReadVariableOp4^generator_2/batch_normalization_22/ReadVariableOp_1C^generator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpE^generator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12^generator_2/batch_normalization_23/ReadVariableOp4^generator_2/batch_normalization_23/ReadVariableOp_1,^generator_2/conv2d_25/Conv2D/ReadVariableOp,^generator_2/conv2d_26/Conv2D/ReadVariableOp,^generator_2/conv2d_27/Conv2D/ReadVariableOp,^generator_2/conv2d_28/Conv2D/ReadVariableOp-^generator_2/conv2d_29/BiasAdd/ReadVariableOp,^generator_2/conv2d_29/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2ї
Dgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1Dgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12Z
+generator_2/conv2d_25/Conv2D/ReadVariableOp+generator_2/conv2d_25/Conv2D/ReadVariableOp2j
3generator_2/batch_normalization_22/ReadVariableOp_13generator_2/batch_normalization_22/ReadVariableOp_12Z
+generator_2/conv2d_29/Conv2D/ReadVariableOp+generator_2/conv2d_29/Conv2D/ReadVariableOp2f
1generator_2/batch_normalization_22/ReadVariableOp1generator_2/batch_normalization_22/ReadVariableOp2ї
Dgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1Dgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12j
3generator_2/batch_normalization_23/ReadVariableOp_13generator_2/batch_normalization_23/ReadVariableOp_12Z
+generator_2/conv2d_26/Conv2D/ReadVariableOp+generator_2/conv2d_26/Conv2D/ReadVariableOp2ѕ
Bgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOpBgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2ѕ
Bgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOpBgenerator_2/batch_normalization_21/FusedBatchNormV3/ReadVariableOp2ѕ
Bgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOpBgenerator_2/batch_normalization_22/FusedBatchNormV3/ReadVariableOp2ѕ
Bgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOpBgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp2f
1generator_2/batch_normalization_20/ReadVariableOp1generator_2/batch_normalization_20/ReadVariableOp2ї
Dgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Dgenerator_2/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12Z
+generator_2/conv2d_27/Conv2D/ReadVariableOp+generator_2/conv2d_27/Conv2D/ReadVariableOp2f
1generator_2/batch_normalization_23/ReadVariableOp1generator_2/batch_normalization_23/ReadVariableOp2\
,generator_2/conv2d_29/BiasAdd/ReadVariableOp,generator_2/conv2d_29/BiasAdd/ReadVariableOp2j
3generator_2/batch_normalization_20/ReadVariableOp_13generator_2/batch_normalization_20/ReadVariableOp_12ї
Dgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1Dgenerator_2/batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12f
1generator_2/batch_normalization_21/ReadVariableOp1generator_2/batch_normalization_21/ReadVariableOp2Z
+generator_2/conv2d_28/Conv2D/ReadVariableOp+generator_2/conv2d_28/Conv2D/ReadVariableOp2j
3generator_2/batch_normalization_21/ReadVariableOp_13generator_2/batch_normalization_21/ReadVariableOp_1: : : : : : : : :' #
!
_user_specified_name	input_7:'#
!
_user_specified_name	input_8: : : : : : : :	 :
 : : : : : 
▓/
Ў
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634408

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:         B@:@:@:@:@:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
dtype0*
_output_shapes
: *
valueB
 *цp}?║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpЙ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@*
T0х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*>
_input_shapes-
+:         B@::::2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp:& "
 
_user_specified_nameinputs: : : : 
┌
H
,__inference_softmax_2_layer_call_fn_10634544

inputs
identityг
PartitionedCallPartitionedCallinputs*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633180*P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_10633174h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*.
_input_shapes
:         @	:& "
 
_user_specified_nameinputs
█
э
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632813

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*K
_output_shapes9
7:         H@:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*>
_input_shapes-
+:         H@::::2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp: : :& "
 
_user_specified_nameinputs: : 
х
ѓ
9__inference_batch_normalization_23_layer_call_fn_10634448

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633135*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633122*
Tout
2і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         B@*
T0"
identityIdentity:output:0*>
_input_shapes-
+:         B@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
█
э
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633122

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         B@:@:@:@:@:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*/
_output_shapes
:         B@*
T0"
identityIdentity:output:0*>
_input_shapes-
+:         B@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
█
э
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634254

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*K
_output_shapes9
7:         D@:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         D@"
identityIdentity:output:0*>
_input_shapes-
+:         D@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
л
б
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpф
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:	@*
dtype0г
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+                           @*
T0*
strides
*
paddingVALIDЅ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*D
_input_shapes3
1:+                           	:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: 
█
э
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634154

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*K
_output_shapes9
7:         F@:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*>
_input_shapes-
+:         F@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1: : : : :& "
 
_user_specified_nameinputs
вѓ
╩
I__inference_generator_2_layer_call_and_return_conditional_losses_10633762
inputs_0
inputs_1,
(conv2d_25_conv2d_readvariableop_resource2
.batch_normalization_20_readvariableop_resource4
0batch_normalization_20_readvariableop_1_resourceC
?batch_normalization_20_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_26_conv2d_readvariableop_resource2
.batch_normalization_21_readvariableop_resource4
0batch_normalization_21_readvariableop_1_resourceC
?batch_normalization_21_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_27_conv2d_readvariableop_resource2
.batch_normalization_22_readvariableop_resource4
0batch_normalization_22_readvariableop_1_resourceC
?batch_normalization_22_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_28_conv2d_readvariableop_resource2
.batch_normalization_23_readvariableop_resource4
0batch_normalization_23_readvariableop_1_resourceC
?batch_normalization_23_fusedbatchnormv3_readvariableop_resourceE
Abatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource,
(conv2d_29_conv2d_readvariableop_resource-
)conv2d_29_biasadd_readvariableop_resource
identityѕб6batch_normalization_20/FusedBatchNormV3/ReadVariableOpб8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_20/ReadVariableOpб'batch_normalization_20/ReadVariableOp_1б6batch_normalization_21/FusedBatchNormV3/ReadVariableOpб8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_21/ReadVariableOpб'batch_normalization_21/ReadVariableOp_1б6batch_normalization_22/FusedBatchNormV3/ReadVariableOpб8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_22/ReadVariableOpб'batch_normalization_22/ReadVariableOp_1б6batch_normalization_23/FusedBatchNormV3/ReadVariableOpб8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1б%batch_normalization_23/ReadVariableOpб'batch_normalization_23/ReadVariableOp_1бconv2d_25/Conv2D/ReadVariableOpбconv2d_26/Conv2D/ReadVariableOpбconv2d_27/Conv2D/ReadVariableOpбconv2d_28/Conv2D/ReadVariableOpб conv2d_29/BiasAdd/ReadVariableOpбconv2d_29/Conv2D/ReadVariableOpј
zero_padding2d_6/Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:Є
zero_padding2d_6/PadPadinputs_0&zero_padding2d_6/Pad/paddings:output:0*
T0*/
_output_shapes
:         J	ј
zero_padding2d_7/Pad/paddingsConst*
dtype0*
_output_shapes

:*9
value0B."                             Є
zero_padding2d_7/PadPadinputs_1&zero_padding2d_7/Pad/paddings:output:0*/
_output_shapes
:         J	*
T0і
	add_2/addAddV2zero_padding2d_6/Pad:output:0zero_padding2d_7/Pad:output:0*
T0*/
_output_shapes
:         J	Й
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:	@х
conv2d_25/Conv2DConv2Dadd_2/add:z:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         H@*
T0*
strides
e
#batch_normalization_20/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: e
#batch_normalization_20/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: Б
!batch_normalization_20/LogicalAnd
LogicalAnd,batch_normalization_20/LogicalAnd/x:output:0,batch_normalization_20/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Я
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@С
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╝
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3conv2d_25/Conv2D:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         H@:@:@:@:@:*
T0*
U0a
batch_normalization_20/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Ѓ
leaky_re_lu_20/LeakyRelu	LeakyRelu+batch_normalization_20/FusedBatchNormV3:y:0*/
_output_shapes
:         H@Й
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@╬
conv2d_26/Conv2DConv2D&leaky_re_lu_20/LeakyRelu:activations:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         F@*
T0*
strides
e
#batch_normalization_21/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: e
#batch_normalization_21/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: Б
!batch_normalization_21/LogicalAnd
LogicalAnd,batch_normalization_21/LogicalAnd/x:output:0,batch_normalization_21/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_21/ReadVariableOpReadVariableOp.batch_normalization_21_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_21/ReadVariableOp_1ReadVariableOp0batch_normalization_21_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Я
6batch_normalization_21/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_21_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@С
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_21_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╝
'batch_normalization_21/FusedBatchNormV3FusedBatchNormV3conv2d_26/Conv2D:output:0-batch_normalization_21/ReadVariableOp:value:0/batch_normalization_21/ReadVariableOp_1:value:0>batch_normalization_21/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         F@:@:@:@:@:a
batch_normalization_21/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Ѓ
leaky_re_lu_21/LeakyRelu	LeakyRelu+batch_normalization_21/FusedBatchNormV3:y:0*/
_output_shapes
:         F@Й
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@╬
conv2d_27/Conv2DConv2D&leaky_re_lu_21/LeakyRelu:activations:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
strides
*
paddingVALID*/
_output_shapes
:         D@*
T0e
#batch_normalization_22/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: e
#batch_normalization_22/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: Б
!batch_normalization_22/LogicalAnd
LogicalAnd,batch_normalization_22/LogicalAnd/x:output:0,batch_normalization_22/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_22/ReadVariableOpReadVariableOp.batch_normalization_22_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_22/ReadVariableOp_1ReadVariableOp0batch_normalization_22_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Я
6batch_normalization_22/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_22_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@С
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_22_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╝
'batch_normalization_22/FusedBatchNormV3FusedBatchNormV3conv2d_27/Conv2D:output:0-batch_normalization_22/ReadVariableOp:value:0/batch_normalization_22/ReadVariableOp_1:value:0>batch_normalization_22/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*
is_training( *
epsilon%oЃ:*K
_output_shapes9
7:         D@:@:@:@:@:a
batch_normalization_22/ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0Ѓ
leaky_re_lu_22/LeakyRelu	LeakyRelu+batch_normalization_22/FusedBatchNormV3:y:0*/
_output_shapes
:         D@Й
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*&
_output_shapes
:@@*
dtype0╬
conv2d_28/Conv2DConv2D&leaky_re_lu_22/LeakyRelu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*/
_output_shapes
:         B@*
T0*
strides
*
paddingVALIDe
#batch_normalization_23/LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: e
#batch_normalization_23/LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: Б
!batch_normalization_23/LogicalAnd
LogicalAnd,batch_normalization_23/LogicalAnd/x:output:0,batch_normalization_23/LogicalAnd/y:output:0*
_output_shapes
: Й
%batch_normalization_23/ReadVariableOpReadVariableOp.batch_normalization_23_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@┬
'batch_normalization_23/ReadVariableOp_1ReadVariableOp0batch_normalization_23_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Я
6batch_normalization_23/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_23_fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@С
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_23_fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╝
'batch_normalization_23/FusedBatchNormV3FusedBatchNormV3conv2d_28/Conv2D:output:0-batch_normalization_23/ReadVariableOp:value:0/batch_normalization_23/ReadVariableOp_1:value:0>batch_normalization_23/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         B@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:a
batch_normalization_23/ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Ѓ
leaky_re_lu_23/LeakyRelu	LeakyRelu+batch_normalization_23/FusedBatchNormV3:y:0*/
_output_shapes
:         B@Й
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@	╬
conv2d_29/Conv2DConv2D&leaky_re_lu_23/LeakyRelu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
paddingVALID*/
_output_shapes
:         @	*
T0*
strides
┤
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:	Џ
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @	r
softmax_2/SoftmaxSoftmaxconv2d_29/BiasAdd:output:0*/
_output_shapes
:         @	*
T0s
	add_3/addAddV2softmax_2/Softmax:softmax:0inputs_1*/
_output_shapes
:         @	*
T0┬
IdentityIdentityadd_3/add:z:07^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_17^batch_normalization_21/FusedBatchNormV3/ReadVariableOp9^batch_normalization_21/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_21/ReadVariableOp(^batch_normalization_21/ReadVariableOp_17^batch_normalization_22/FusedBatchNormV3/ReadVariableOp9^batch_normalization_22/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_22/ReadVariableOp(^batch_normalization_22/ReadVariableOp_17^batch_normalization_23/FusedBatchNormV3/ReadVariableOp9^batch_normalization_23/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_23/ReadVariableOp(^batch_normalization_23/ReadVariableOp_1 ^conv2d_25/Conv2D/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2N
%batch_normalization_21/ReadVariableOp%batch_normalization_21/ReadVariableOp2R
'batch_normalization_22/ReadVariableOp_1'batch_normalization_22/ReadVariableOp_12t
8batch_normalization_22/FusedBatchNormV3/ReadVariableOp_18batch_normalization_22/FusedBatchNormV3/ReadVariableOp_12B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp2R
'batch_normalization_23/ReadVariableOp_1'batch_normalization_23/ReadVariableOp_12B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2t
8batch_normalization_21/FusedBatchNormV3/ReadVariableOp_18batch_normalization_21/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_22/ReadVariableOp%batch_normalization_22/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2p
6batch_normalization_21/FusedBatchNormV3/ReadVariableOp6batch_normalization_21/FusedBatchNormV3/ReadVariableOp2p
6batch_normalization_22/FusedBatchNormV3/ReadVariableOp6batch_normalization_22/FusedBatchNormV3/ReadVariableOp2p
6batch_normalization_23/FusedBatchNormV3/ReadVariableOp6batch_normalization_23/FusedBatchNormV3/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2N
%batch_normalization_23/ReadVariableOp%batch_normalization_23/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2t
8batch_normalization_23/FusedBatchNormV3/ReadVariableOp_18batch_normalization_23/FusedBatchNormV3/ReadVariableOp_12R
'batch_normalization_21/ReadVariableOp_1'batch_normalization_21/ReadVariableOp_1:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : : : : : : : : : : : : : 
С
M
1__inference_leaky_re_lu_22_layer_call_fn_10634358

inputs
identity▒
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-10633056*U
fPRN
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10633050*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         D@"
identityIdentity:output:0*.
_input_shapes
:         D@:& "
 
_user_specified_nameinputs
в
ѓ
9__inference_batch_normalization_22_layer_call_fn_10634348

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*A
_output_shapes/
-:+                           @*/
_gradient_op_typePartitionedCall-10632534*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632533ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
█
э
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633978

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*K
_output_shapes9
7:         H@:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*>
_input_shapes-
+:         H@::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
С
o
C__inference_add_3_layer_call_and_return_conditional_losses_10634550
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:         @	W
IdentityIdentityadd:z:0*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*I
_input_shapes8
6:         @	:         @	:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
л
б
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpф
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@г
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+                           @*
T0*
strides
*
paddingVALIDЅ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: 
Љ
э
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633902

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
▓/
Ў
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634232

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*K
_output_shapes9
7:         D@:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpЙ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpУ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         D@"
identityIdentity:output:0*>
_input_shapes-
+:         D@::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:& "
 
_user_specified_nameinputs: : : : 
Љ
э
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634330

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z *
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
is_training( J
ConstConst*
dtype0*
_output_shapes
: *
valueB
 *цp}?Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
ЄZ
К
$__inference__traced_restore_10634727
file_prefix%
!assignvariableop_conv2d_25_kernel3
/assignvariableop_1_batch_normalization_20_gamma2
.assignvariableop_2_batch_normalization_20_beta9
5assignvariableop_3_batch_normalization_20_moving_mean=
9assignvariableop_4_batch_normalization_20_moving_variance'
#assignvariableop_5_conv2d_26_kernel3
/assignvariableop_6_batch_normalization_21_gamma2
.assignvariableop_7_batch_normalization_21_beta9
5assignvariableop_8_batch_normalization_21_moving_mean=
9assignvariableop_9_batch_normalization_21_moving_variance(
$assignvariableop_10_conv2d_27_kernel4
0assignvariableop_11_batch_normalization_22_gamma3
/assignvariableop_12_batch_normalization_22_beta:
6assignvariableop_13_batch_normalization_22_moving_mean>
:assignvariableop_14_batch_normalization_22_moving_variance(
$assignvariableop_15_conv2d_28_kernel4
0assignvariableop_16_batch_normalization_23_gamma3
/assignvariableop_17_batch_normalization_23_beta:
6assignvariableop_18_batch_normalization_23_moving_mean>
:assignvariableop_19_batch_normalization_23_moving_variance(
$assignvariableop_20_conv2d_29_kernel&
"assignvariableop_21_conv2d_29_bias
identity_23ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1ж

RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*Ј

valueЁ
Bѓ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEю
RestoreV2/shape_and_slicesConst"/device:CPU:0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:ї
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0}
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_25_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0Ј
AssignVariableOp_1AssignVariableOp/assignvariableop_1_batch_normalization_20_gammaIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:ј
AssignVariableOp_2AssignVariableOp.assignvariableop_2_batch_normalization_20_betaIdentity_2:output:0*
_output_shapes
 *
dtype0N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:Ћ
AssignVariableOp_3AssignVariableOp5assignvariableop_3_batch_normalization_20_moving_meanIdentity_3:output:0*
_output_shapes
 *
dtype0N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:Ў
AssignVariableOp_4AssignVariableOp9assignvariableop_4_batch_normalization_20_moving_varianceIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:Ѓ
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv2d_26_kernelIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:Ј
AssignVariableOp_6AssignVariableOp/assignvariableop_6_batch_normalization_21_gammaIdentity_6:output:0*
_output_shapes
 *
dtype0N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:ј
AssignVariableOp_7AssignVariableOp.assignvariableop_7_batch_normalization_21_betaIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:Ћ
AssignVariableOp_8AssignVariableOp5assignvariableop_8_batch_normalization_21_moving_meanIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:Ў
AssignVariableOp_9AssignVariableOp9assignvariableop_9_batch_normalization_21_moving_varianceIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:є
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_27_kernelIdentity_10:output:0*
_output_shapes
 *
dtype0P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:њ
AssignVariableOp_11AssignVariableOp0assignvariableop_11_batch_normalization_22_gammaIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0Љ
AssignVariableOp_12AssignVariableOp/assignvariableop_12_batch_normalization_22_betaIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:ў
AssignVariableOp_13AssignVariableOp6assignvariableop_13_batch_normalization_22_moving_meanIdentity_13:output:0*
_output_shapes
 *
dtype0P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:ю
AssignVariableOp_14AssignVariableOp:assignvariableop_14_batch_normalization_22_moving_varianceIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:є
AssignVariableOp_15AssignVariableOp$assignvariableop_15_conv2d_28_kernelIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:њ
AssignVariableOp_16AssignVariableOp0assignvariableop_16_batch_normalization_23_gammaIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:Љ
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_23_betaIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:ў
AssignVariableOp_18AssignVariableOp6assignvariableop_18_batch_normalization_23_moving_meanIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:ю
AssignVariableOp_19AssignVariableOp:assignvariableop_19_batch_normalization_23_moving_varianceIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:є
AssignVariableOp_20AssignVariableOp$assignvariableop_20_conv2d_29_kernelIdentity_20:output:0*
dtype0*
_output_shapes
 P
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:ё
AssignVariableOp_21AssignVariableOp"assignvariableop_21_conv2d_29_biasIdentity_21:output:0*
dtype0*
_output_shapes
 ї
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:х
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 │
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: └
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_23Identity_23:output:0*m
_input_shapes\
Z: ::::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122
RestoreV2_1RestoreV2_12*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : 
У/
Ў
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634056

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 ZN
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
dtype0*
_output_shapes
: *
valueB J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_1: : : :& "
 
_user_specified_nameinputs: 
█
э
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634430

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
dtype0
*
_output_shapes
: *
value	B
 Z^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         B@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*>
_input_shapes-
+:         B@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
║
O
3__inference_zero_padding2d_6_layer_call_fn_10632035

inputs
identity╬
PartitionedCallPartitionedCallinputs*W
fRRP
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*J
_output_shapes8
6:4                                    */
_gradient_op_typePartitionedCall-10632032Ѓ
IdentityIdentityPartitionedCall:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
▓/
Ў
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632791

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*K
_output_shapes9
7:         H@:@:@:@:@:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 У
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*>
_input_shapes-
+:         H@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp: : : :& "
 
_user_specified_nameinputs: 
▄
m
C__inference_add_2_layer_call_and_return_conditional_losses_10632740

inputs
inputs_1
identityX
addAddV2inputsinputs_1*
T0*/
_output_shapes
:         J	W
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:         J	"
identityIdentity:output:0*I
_input_shapes8
6:         J	:         J	:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
У/
Ў
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634308

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: : :& "
 
_user_specified_nameinputs: : 
Ч
Њ
&__inference_signature_wrapper_10633492
input_7
input_8"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identityѕбStatefulPartitionedCallЄ
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*#
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633467*,
f'R%
#__inference__wrapped_model_10632017*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_7:'#
!
_user_specified_name	input_8: : : : : : : :	 :
 : : : : : : : : : : : : : 
Љ
э
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632371

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0J
ConstConst*
_output_shapes
: *
valueB
 *цp}?*
dtype0Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
х
ѓ
9__inference_batch_normalization_21_layer_call_fn_10634172

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632916*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632929і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*>
_input_shapes-
+:         F@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
У/
Ў
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10632661

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
dtype0*
_output_shapes
: *
valueB J
Const_1Const*
_output_shapes
: *
valueB *
dtype0Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
_output_shapes
: *
valueB
 *цp}?*
dtype0║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
: ┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp: : : : :& "
 
_user_specified_nameinputs
С
Ѕ
,__inference_conv2d_27_layer_call_fn_10632397

inputs"
statefulpartitionedcall_args_1
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*A
_output_shapes/
-:+                           @*
Tin
2*/
_gradient_op_typePartitionedCall-10632393*P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
л
б
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225

inputs"
conv2d_readvariableop_resource
identityѕбConv2D/ReadVariableOpф
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*&
_output_shapes
:@@г
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*A
_output_shapes/
-:+                           @*
T0*
strides
*
paddingVALIDЅ
IdentityIdentityConv2D:output:0^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*D
_input_shapes3
1:+                           @:2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs: 
ф
Џ
.__inference_generator_2_layer_call_fn_10633330
input_7
input_8"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*/
_output_shapes
:         @	*#
Tin
2*/
_gradient_op_typePartitionedCall-10633305*R
fMRK
I__inference_generator_2_layer_call_and_return_conditional_losses_10633304*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_7:'#
!
_user_specified_name	input_8: : : : : : : :	 :
 : : : : : : : : : : : : : 
У/
Ў
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632175

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
valueB *
dtype0*
_output_shapes
: Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@*
T0Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
_output_shapes
:@*
T0─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpх
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0*
_output_shapes
 Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp: : : : :& "
 
_user_specified_nameinputs
в
ѓ
9__inference_batch_normalization_20_layer_call_fn_10633911

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*A
_output_shapes/
-:+                           @*/
_gradient_op_typePartitionedCall-10632176*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632175*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
Љ
э
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632209

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1: :& "
 
_user_specified_nameinputs: : : 
У/
Ў
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633880

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z*
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
_output_shapes
: *
valueB *
dtype0Ѓ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp█
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpФ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOpЙ
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
: *
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@р
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
dtype0Щ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*A
_output_shapes/
-:+                           @*
T0"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp: : : :& "
 
_user_specified_nameinputs: 
Џ
h
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10634001

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         H@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*.
_input_shapes
:         H@:& "
 
_user_specified_nameinputs
░
Ю
.__inference_generator_2_layer_call_fn_10633818
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*R
fMRK
I__inference_generator_2_layer_call_and_return_conditional_losses_10633380*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         @	*#
Tin
2*/
_gradient_op_typePartitionedCall-10633381і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : : : : : : : : : : : : : 
в
ѓ
9__inference_batch_normalization_22_layer_call_fn_10634339

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*A
_output_shapes/
-:+                           @*
Tin	
2*/
_gradient_op_typePartitionedCall-10632500*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632499*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
С
M
1__inference_leaky_re_lu_21_layer_call_fn_10634182

inputs
identity▒
PartitionedCallPartitionedCallinputs*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632953*U
fPRN
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10632947*
Tout
2h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*.
_input_shapes
:         F@:& "
 
_user_specified_nameinputs
░
Ю
.__inference_generator_2_layer_call_fn_10633790
inputs_0
inputs_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identityѕбStatefulPartitionedCall»
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*/
_gradient_op_typePartitionedCall-10633305*R
fMRK
I__inference_generator_2_layer_call_and_return_conditional_losses_10633304*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         @	*#
Tin
2і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1: : : : : : : :	 :
 : : : : : : : : : : : : : 
Љ
э
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634506

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
dtype0
*
_output_shapes
: *
value	B
 Z N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
│
Г
,__inference_conv2d_29_layer_call_fn_10632725

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-10632720*P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*A
_output_shapes/
-:+                           	ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           	"
identityIdentity:output:0*H
_input_shapes7
5:+                           @::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
Љ
э
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632533

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
is_training( *
epsilon%oЃ:*]
_output_shapesK
I:+                           @:@:@:@:@:*
T0*
U0J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: Я
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs: : : : 
ф
Џ
.__inference_generator_2_layer_call_fn_10633406
input_7
input_8"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23
identityѕбStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinput_7input_8statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*#
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633381*R
fMRK
I__inference_generator_2_layer_call_and_return_conditional_losses_10633380і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:' #
!
_user_specified_name	input_7:'#
!
_user_specified_name	input_8: : : : : : : :	 :
 : : : : : : : : : : : : : 
СV
▓
I__inference_generator_2_layer_call_and_return_conditional_losses_10633208
input_7
input_8,
(conv2d_25_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_29
5batch_normalization_20_statefulpartitionedcall_args_39
5batch_normalization_20_statefulpartitionedcall_args_4,
(conv2d_26_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_29
5batch_normalization_21_statefulpartitionedcall_args_39
5batch_normalization_21_statefulpartitionedcall_args_4,
(conv2d_27_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_29
5batch_normalization_22_statefulpartitionedcall_args_39
5batch_normalization_22_statefulpartitionedcall_args_4,
(conv2d_28_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_29
5batch_normalization_23_statefulpartitionedcall_args_39
5batch_normalization_23_statefulpartitionedcall_args_4,
(conv2d_29_statefulpartitionedcall_args_1,
(conv2d_29_statefulpartitionedcall_args_2
identityѕб.batch_normalization_20/StatefulPartitionedCallб.batch_normalization_21/StatefulPartitionedCallб.batch_normalization_22/StatefulPartitionedCallб.batch_normalization_23/StatefulPartitionedCallб!conv2d_25/StatefulPartitionedCallб!conv2d_26/StatefulPartitionedCallб!conv2d_27/StatefulPartitionedCallб!conv2d_28/StatefulPartitionedCallб!conv2d_29/StatefulPartitionedCall┼
 zero_padding2d_6/PartitionedCallPartitionedCallinput_7*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         J	*/
_gradient_op_typePartitionedCall-10632032*W
fRRP
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026┼
 zero_padding2d_7/PartitionedCallPartitionedCallinput_8*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         J	*/
_gradient_op_typePartitionedCall-10632050*W
fRRP
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044*
Tout
2§
add_2/PartitionedCallPartitionedCall)zero_padding2d_6/PartitionedCall:output:0)zero_padding2d_7/PartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         J	*/
_gradient_op_typePartitionedCall-10632747*L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_10632740*
Tout
2Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         H@*/
_gradient_op_typePartitionedCall-10632069*P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063*
Tout
2С
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:05batch_normalization_20_statefulpartitionedcall_args_15batch_normalization_20_statefulpartitionedcall_args_25batch_normalization_20_statefulpartitionedcall_args_35batch_normalization_20_statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10632816*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632791*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin	
2ы
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2*/
_gradient_op_typePartitionedCall-10632850*U
fPRN
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10632844*
Tout
2њ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1*P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         F@*
Tin
2*/
_gradient_op_typePartitionedCall-10632231С
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:05batch_normalization_21_statefulpartitionedcall_args_15batch_normalization_21_statefulpartitionedcall_args_25batch_normalization_21_statefulpartitionedcall_args_35batch_normalization_21_statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10632919*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632894*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         F@*
Tin	
2ы
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*U
fPRN
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10632947*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632953њ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0(conv2d_27_statefulpartitionedcall_args_1*
Tin
2*/
_output_shapes
:         D@*/
_gradient_op_typePartitionedCall-10632393*P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8С
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:05batch_normalization_22_statefulpartitionedcall_args_15batch_normalization_22_statefulpartitionedcall_args_25batch_normalization_22_statefulpartitionedcall_args_35batch_normalization_22_statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin	
2*/
_gradient_op_typePartitionedCall-10633022*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632997*
Tout
2ы
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:         D@*/
_gradient_op_typePartitionedCall-10633056*U
fPRN
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10633050*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8њ
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0(conv2d_28_statefulpartitionedcall_args_1*/
_gradient_op_typePartitionedCall-10632555*P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         B@С
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:05batch_normalization_23_statefulpartitionedcall_args_15batch_normalization_23_statefulpartitionedcall_args_25batch_normalization_23_statefulpartitionedcall_args_35batch_normalization_23_statefulpartitionedcall_args_4*
Tin	
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633125*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633100*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8ы
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*U
fPRN
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10633153*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633159й
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0(conv2d_29_statefulpartitionedcall_args_1(conv2d_29_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-10632720*P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         @	*
Tin
2┌
softmax_2/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633180*P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_10633174*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8н
add_3/PartitionedCallPartitionedCall"softmax_2/PartitionedCall:output:0input_8*L
fGRE
C__inference_add_3_layer_call_and_return_conditional_losses_10633193*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633200Т
IdentityIdentityadd_3/PartitionedCall:output:0/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall:' #
!
_user_specified_name	input_7:'#
!
_user_specified_name	input_8: : : : : : : :	 :
 : : : : : : : : : : : : : 
в
ѓ
9__inference_batch_normalization_20_layer_call_fn_10633920

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10632210*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632209*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*A
_output_shapes/
-:+                           @*
Tin	
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
Ћ
c
G__inference_softmax_2_layer_call_and_return_conditional_losses_10634539

inputs
identityT
SoftmaxSoftmaxinputs*
T0*/
_output_shapes
:         @	a
IdentityIdentitySoftmax:softmax:0*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*.
_input_shapes
:         @	:& "
 
_user_specified_nameinputs
▓/
Ў
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632997

inputs
readvariableop_resource
readvariableop_1_resource0
,assignmovingavg_read_readvariableop_resource2
.assignmovingavg_1_read_readvariableop_resource
identityѕб#AssignMovingAvg/AssignSubVariableOpб#AssignMovingAvg/Read/ReadVariableOpбAssignMovingAvg/ReadVariableOpб%AssignMovingAvg_1/AssignSubVariableOpб%AssignMovingAvg_1/Read/ReadVariableOpб AssignMovingAvg_1/ReadVariableOpбReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
_output_shapes
: *
value	B
 Z*
dtype0
N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0H
ConstConst*
valueB *
dtype0*
_output_shapes
: J
Const_1Const*
dtype0*
_output_shapes
: *
valueB ы
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*K
_output_shapes9
7:         D@:@:@:@:@:*
T0*
U0*
epsilon%oЃ:L
Const_2Const*
valueB
 *цp}?*
dtype0*
_output_shapes
: ║
#AssignMovingAvg/Read/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@v
AssignMovingAvg/IdentityIdentity+AssignMovingAvg/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@└
AssignMovingAvg/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
valueB
 *  ђ?*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
: М
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
: █
AssignMovingAvg/ReadVariableOpReadVariableOp,assignmovingavg_read_readvariableop_resource$^AssignMovingAvg/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Ь
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@┘
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
_output_shapes
:@Ф
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp,assignmovingavg_read_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*6
_class,
*(loc:@AssignMovingAvg/Read/ReadVariableOp*
dtype0*
_output_shapes
 Й
%AssignMovingAvg_1/Read/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
dtype0z
AssignMovingAvg_1/IdentityIdentity-AssignMovingAvg_1/Read/ReadVariableOp:value:0*
T0*
_output_shapes
:@─
AssignMovingAvg_1/sub/xConst",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp┘
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
: р
 AssignMovingAvg_1/ReadVariableOpReadVariableOp.assignmovingavg_1_read_readvariableop_resource&^AssignMovingAvg_1/Read/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Э
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0",/job:localhost/replica:0/task:0/device:GPU:0*
_output_shapes
:@*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpр
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0",/job:localhost/replica:0/task:0/device:GPU:0*
T0*8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOp*
_output_shapes
:@х
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp.assignmovingavg_1_read_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
 *8
_class.
,*loc:@AssignMovingAvg_1/Read/ReadVariableOpУ
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp$^AssignMovingAvg/Read/ReadVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp&^AssignMovingAvg_1/Read/ReadVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         D@"
identityIdentity:output:0*>
_input_shapes-
+:         D@::::2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2$
ReadVariableOp_1ReadVariableOp_12J
#AssignMovingAvg/Read/ReadVariableOp#AssignMovingAvg/Read/ReadVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/Read/ReadVariableOp%AssignMovingAvg_1/Read/ReadVariableOp2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
Х
T
(__inference_add_3_layer_call_fn_10634556
inputs_0
inputs_1
identityх
PartitionedCallPartitionedCallinputs_0inputs_1*/
_gradient_op_typePartitionedCall-10633200*L
fGRE
C__inference_add_3_layer_call_and_return_conditional_losses_10633193*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:         @	*
T0"
identityIdentity:output:0*I
_input_shapes8
6:         @	:         @	:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
Х
T
(__inference_add_2_layer_call_fn_10633830
inputs_0
inputs_1
identityх
PartitionedCallPartitionedCallinputs_0inputs_1*L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_10632740*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         J	*
Tin
2*/
_gradient_op_typePartitionedCall-10632747h
IdentityIdentityPartitionedCall:output:0*/
_output_shapes
:         J	*
T0"
identityIdentity:output:0*I
_input_shapes8
6:         J	:         J	:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
х
ѓ
9__inference_batch_normalization_21_layer_call_fn_10634163

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632894*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632919і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*/
_output_shapes
:         F@*
T0"
identityIdentity:output:0*>
_input_shapes-
+:         F@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
С
o
C__inference_add_2_layer_call_and_return_conditional_losses_10633824
inputs_0
inputs_1
identityZ
addAddV2inputs_0inputs_1*/
_output_shapes
:         J	*
T0W
IdentityIdentityadd:z:0*/
_output_shapes
:         J	*
T0"
identityIdentity:output:0*I
_input_shapes8
6:         J	:         J	:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
С
Ѕ
,__inference_conv2d_25_layer_call_fn_10632073

inputs"
statefulpartitionedcall_args_1
identityѕбStatefulPartitionedCall№
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*A
_output_shapes/
-:+                           @*/
_gradient_op_typePartitionedCall-10632069*P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063*
Tout
2ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*D
_input_shapes3
1:+                           	:22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: 
║
O
3__inference_zero_padding2d_7_layer_call_fn_10632053

inputs
identity╬
PartitionedCallPartitionedCallinputs*/
_gradient_op_typePartitionedCall-10632050*W
fRRP
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*J
_output_shapes8
6:4                                    *
Tin
2Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
х
ѓ
9__inference_batch_normalization_20_layer_call_fn_10633987

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall═
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin	
2*/
_gradient_op_typePartitionedCall-10632816*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632791*
Tout
2і
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         H@"
identityIdentity:output:0*>
_input_shapes-
+:         H@::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
Џ
h
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10632947

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         F@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*.
_input_shapes
:         F@:& "
 
_user_specified_nameinputs
█
э
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632916

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1N
LogicalAnd/xConst*
value	B
 Z *
dtype0
*
_output_shapes
: N
LogicalAnd/yConst*
value	B
 Z*
dtype0
*
_output_shapes
: ^

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: љ
ReadVariableOpReadVariableOpreadvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@ћ
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@▓
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource",/job:localhost/replica:0/task:0/device:GPU:0*
dtype0*
_output_shapes
:@Х
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*K
_output_shapes9
7:         F@:@:@:@:@:*
T0*
U0*
is_training( *
epsilon%oЃ:J
ConstConst*
valueB
 *цp}?*
dtype0*
_output_shapes
: ╬
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:         F@"
identityIdentity:output:0*>
_input_shapes-
+:         F@::::2 
ReadVariableOpReadVariableOp2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_1: : :& "
 
_user_specified_nameinputs: : 
сV
▓
I__inference_generator_2_layer_call_and_return_conditional_losses_10633380

inputs
inputs_1,
(conv2d_25_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_29
5batch_normalization_20_statefulpartitionedcall_args_39
5batch_normalization_20_statefulpartitionedcall_args_4,
(conv2d_26_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_29
5batch_normalization_21_statefulpartitionedcall_args_39
5batch_normalization_21_statefulpartitionedcall_args_4,
(conv2d_27_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_29
5batch_normalization_22_statefulpartitionedcall_args_39
5batch_normalization_22_statefulpartitionedcall_args_4,
(conv2d_28_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_29
5batch_normalization_23_statefulpartitionedcall_args_39
5batch_normalization_23_statefulpartitionedcall_args_4,
(conv2d_29_statefulpartitionedcall_args_1,
(conv2d_29_statefulpartitionedcall_args_2
identityѕб.batch_normalization_20/StatefulPartitionedCallб.batch_normalization_21/StatefulPartitionedCallб.batch_normalization_22/StatefulPartitionedCallб.batch_normalization_23/StatefulPartitionedCallб!conv2d_25/StatefulPartitionedCallб!conv2d_26/StatefulPartitionedCallб!conv2d_27/StatefulPartitionedCallб!conv2d_28/StatefulPartitionedCallб!conv2d_29/StatefulPartitionedCall─
 zero_padding2d_6/PartitionedCallPartitionedCallinputs*
Tin
2*/
_output_shapes
:         J	*/
_gradient_op_typePartitionedCall-10632032*W
fRRP
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8к
 zero_padding2d_7/PartitionedCallPartitionedCallinputs_1*/
_gradient_op_typePartitionedCall-10632050*W
fRRP
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         J	*
Tin
2§
add_2/PartitionedCallPartitionedCall)zero_padding2d_6/PartitionedCall:output:0)zero_padding2d_7/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10632747*L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_10632740*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         J	*
Tin
2Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2*/
_gradient_op_typePartitionedCall-10632069*P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063*
Tout
2С
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:05batch_normalization_20_statefulpartitionedcall_args_15batch_normalization_20_statefulpartitionedcall_args_25batch_normalization_20_statefulpartitionedcall_args_35batch_normalization_20_statefulpartitionedcall_args_4*/
_output_shapes
:         H@*
Tin	
2*/
_gradient_op_typePartitionedCall-10632826*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632813*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8ы
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2*/
_gradient_op_typePartitionedCall-10632850*U
fPRN
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10632844*
Tout
2њ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1*
Tin
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632231*P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8С
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:05batch_normalization_21_statefulpartitionedcall_args_15batch_normalization_21_statefulpartitionedcall_args_25batch_normalization_21_statefulpartitionedcall_args_35batch_normalization_21_statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         F@*
Tin	
2*/
_gradient_op_typePartitionedCall-10632929*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632916*
Tout
2ы
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10632953*U
fPRN
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10632947*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         F@*
Tin
2њ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0(conv2d_27_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin
2*/
_gradient_op_typePartitionedCall-10632393*P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387*
Tout
2С
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:05batch_normalization_22_statefulpartitionedcall_args_15batch_normalization_22_statefulpartitionedcall_args_25batch_normalization_22_statefulpartitionedcall_args_35batch_normalization_22_statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10633032*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10633019*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin	
2ы
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10633056*U
fPRN
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10633050*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin
2њ
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0(conv2d_28_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         B@*
Tin
2*/
_gradient_op_typePartitionedCall-10632555*P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549*
Tout
2С
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:05batch_normalization_23_statefulpartitionedcall_args_15batch_normalization_23_statefulpartitionedcall_args_25batch_normalization_23_statefulpartitionedcall_args_35batch_normalization_23_statefulpartitionedcall_args_4*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633135*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633122ы
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633159*U
fPRN
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10633153*
Tout
2й
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0(conv2d_29_statefulpartitionedcall_args_1(conv2d_29_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-10632720*P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	┌
softmax_2/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         @	*
Tin
2*/
_gradient_op_typePartitionedCall-10633180*P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_10633174*
Tout
2Н
add_3/PartitionedCallPartitionedCall"softmax_2/PartitionedCall:output:0inputs_1*/
_gradient_op_typePartitionedCall-10633200*L
fGRE
C__inference_add_3_layer_call_and_return_conditional_losses_10633193*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	Т
IdentityIdentityadd_3/PartitionedCall:output:0/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall: : : : : : : : : : : : : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : : : : :	 :
 
Б5
┤
!__inference__traced_save_10634648
file_prefix/
+savev2_conv2d_25_kernel_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop;
7savev2_batch_normalization_21_gamma_read_readvariableop:
6savev2_batch_normalization_21_beta_read_readvariableopA
=savev2_batch_normalization_21_moving_mean_read_readvariableopE
Asavev2_batch_normalization_21_moving_variance_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop;
7savev2_batch_normalization_22_gamma_read_readvariableop:
6savev2_batch_normalization_22_beta_read_readvariableopA
=savev2_batch_normalization_22_moving_mean_read_readvariableopE
Asavev2_batch_normalization_22_moving_variance_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop;
7savev2_batch_normalization_23_gamma_read_readvariableop:
6savev2_batch_normalization_23_beta_read_readvariableopA
=savev2_batch_normalization_23_moving_mean_read_readvariableopE
Asavev2_batch_normalization_23_moving_variance_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1ј
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_cf6e3da706164f76b31836ce61a0ae02/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
dtype0*
_output_shapes
: *
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Т

SaveV2/tensor_namesConst"/device:CPU:0*Ј

valueЁ
Bѓ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Ў
SaveV2/shape_and_slicesConst"/device:CPU:0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Ј
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_25_kernel_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop7savev2_batch_normalization_21_gamma_read_readvariableop6savev2_batch_normalization_21_beta_read_readvariableop=savev2_batch_normalization_21_moving_mean_read_readvariableopAsavev2_batch_normalization_21_moving_variance_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop7savev2_batch_normalization_22_gamma_read_readvariableop6savev2_batch_normalization_22_beta_read_readvariableop=savev2_batch_normalization_22_moving_mean_read_readvariableopAsavev2_batch_normalization_22_moving_variance_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop7savev2_batch_normalization_23_gamma_read_readvariableop6savev2_batch_normalization_23_beta_read_readvariableop=savev2_batch_normalization_23_moving_mean_read_readvariableopAsavev2_batch_normalization_23_moving_variance_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *$
dtypes
2h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ќ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:ќ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*┘
_input_shapesК
─: :	@:@:@:@:@:@@:@:@:@:@:@@:@:@:@:@:@@:@:@:@:@:@	:	: 2
SaveV2_1SaveV2_12(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : : : 
сV
▓
I__inference_generator_2_layer_call_and_return_conditional_losses_10633304

inputs
inputs_1,
(conv2d_25_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_29
5batch_normalization_20_statefulpartitionedcall_args_39
5batch_normalization_20_statefulpartitionedcall_args_4,
(conv2d_26_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_29
5batch_normalization_21_statefulpartitionedcall_args_39
5batch_normalization_21_statefulpartitionedcall_args_4,
(conv2d_27_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_29
5batch_normalization_22_statefulpartitionedcall_args_39
5batch_normalization_22_statefulpartitionedcall_args_4,
(conv2d_28_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_29
5batch_normalization_23_statefulpartitionedcall_args_39
5batch_normalization_23_statefulpartitionedcall_args_4,
(conv2d_29_statefulpartitionedcall_args_1,
(conv2d_29_statefulpartitionedcall_args_2
identityѕб.batch_normalization_20/StatefulPartitionedCallб.batch_normalization_21/StatefulPartitionedCallб.batch_normalization_22/StatefulPartitionedCallб.batch_normalization_23/StatefulPartitionedCallб!conv2d_25/StatefulPartitionedCallб!conv2d_26/StatefulPartitionedCallб!conv2d_27/StatefulPartitionedCallб!conv2d_28/StatefulPartitionedCallб!conv2d_29/StatefulPartitionedCall─
 zero_padding2d_6/PartitionedCallPartitionedCallinputs*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         J	*/
_gradient_op_typePartitionedCall-10632032*W
fRRP
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026*
Tout
2к
 zero_padding2d_7/PartitionedCallPartitionedCallinputs_1*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         J	*
Tin
2*/
_gradient_op_typePartitionedCall-10632050*W
fRRP
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044*
Tout
2§
add_2/PartitionedCallPartitionedCall)zero_padding2d_6/PartitionedCall:output:0)zero_padding2d_7/PartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10632747*L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_10632740*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         J	*
Tin
2Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1*/
_gradient_op_typePartitionedCall-10632069*P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2С
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:05batch_normalization_20_statefulpartitionedcall_args_15batch_normalization_20_statefulpartitionedcall_args_25batch_normalization_20_statefulpartitionedcall_args_35batch_normalization_20_statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         H@*/
_gradient_op_typePartitionedCall-10632816*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632791*
Tout
2ы
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2*/
_gradient_op_typePartitionedCall-10632850*U
fPRN
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10632844*
Tout
2њ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1*/
_gradient_op_typePartitionedCall-10632231*P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         F@С
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:05batch_normalization_21_statefulpartitionedcall_args_15batch_normalization_21_statefulpartitionedcall_args_25batch_normalization_21_statefulpartitionedcall_args_35batch_normalization_21_statefulpartitionedcall_args_4*
Tin	
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632919*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632894*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8ы
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632953*U
fPRN
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10632947*
Tout
2њ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0(conv2d_27_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         D@*/
_gradient_op_typePartitionedCall-10632393*P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387*
Tout
2С
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:05batch_normalization_22_statefulpartitionedcall_args_15batch_normalization_22_statefulpartitionedcall_args_25batch_normalization_22_statefulpartitionedcall_args_35batch_normalization_22_statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10633022*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10632997*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         D@ы
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         D@*/
_gradient_op_typePartitionedCall-10633056*U
fPRN
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10633050*
Tout
2њ
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0(conv2d_28_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10632555*P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549*
Tout
2С
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:05batch_normalization_23_statefulpartitionedcall_args_15batch_normalization_23_statefulpartitionedcall_args_25batch_normalization_23_statefulpartitionedcall_args_35batch_normalization_23_statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         B@*/
_gradient_op_typePartitionedCall-10633125*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633100*
Tout
2ы
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10633159*U
fPRN
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10633153*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         B@*
Tin
2й
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0(conv2d_29_statefulpartitionedcall_args_1(conv2d_29_statefulpartitionedcall_args_2*/
_gradient_op_typePartitionedCall-10632720*P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	┌
softmax_2/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_10633174*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         @	*
Tin
2*/
_gradient_op_typePartitionedCall-10633180Н
add_3/PartitionedCallPartitionedCall"softmax_2/PartitionedCall:output:0inputs_1*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633200*L
fGRE
C__inference_add_3_layer_call_and_return_conditional_losses_10633193*
Tout
2Т
IdentityIdentityadd_3/PartitionedCall:output:0/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall: : : :	 :
 : : : : : : : : : : : : : :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs: : : : 
в
ѓ
9__inference_batch_normalization_23_layer_call_fn_10634515

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10632662*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10632661*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*A
_output_shapes/
-:+                           @ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall: : : :& "
 
_user_specified_nameinputs: 
СV
▓
I__inference_generator_2_layer_call_and_return_conditional_losses_10633255
input_7
input_8,
(conv2d_25_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_19
5batch_normalization_20_statefulpartitionedcall_args_29
5batch_normalization_20_statefulpartitionedcall_args_39
5batch_normalization_20_statefulpartitionedcall_args_4,
(conv2d_26_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_19
5batch_normalization_21_statefulpartitionedcall_args_29
5batch_normalization_21_statefulpartitionedcall_args_39
5batch_normalization_21_statefulpartitionedcall_args_4,
(conv2d_27_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_19
5batch_normalization_22_statefulpartitionedcall_args_29
5batch_normalization_22_statefulpartitionedcall_args_39
5batch_normalization_22_statefulpartitionedcall_args_4,
(conv2d_28_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_19
5batch_normalization_23_statefulpartitionedcall_args_29
5batch_normalization_23_statefulpartitionedcall_args_39
5batch_normalization_23_statefulpartitionedcall_args_4,
(conv2d_29_statefulpartitionedcall_args_1,
(conv2d_29_statefulpartitionedcall_args_2
identityѕб.batch_normalization_20/StatefulPartitionedCallб.batch_normalization_21/StatefulPartitionedCallб.batch_normalization_22/StatefulPartitionedCallб.batch_normalization_23/StatefulPartitionedCallб!conv2d_25/StatefulPartitionedCallб!conv2d_26/StatefulPartitionedCallб!conv2d_27/StatefulPartitionedCallб!conv2d_28/StatefulPartitionedCallб!conv2d_29/StatefulPartitionedCall┼
 zero_padding2d_6/PartitionedCallPartitionedCallinput_7*/
_gradient_op_typePartitionedCall-10632032*W
fRRP
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         J	*
Tin
2┼
 zero_padding2d_7/PartitionedCallPartitionedCallinput_8*/
_gradient_op_typePartitionedCall-10632050*W
fRRP
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         J	§
add_2/PartitionedCallPartitionedCall)zero_padding2d_6/PartitionedCall:output:0)zero_padding2d_7/PartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         J	*/
_gradient_op_typePartitionedCall-10632747*L
fGRE
C__inference_add_2_layer_call_and_return_conditional_losses_10632740*
Tout
2Ѕ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0(conv2d_25_statefulpartitionedcall_args_1*P
fKRI
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2*/
_gradient_op_typePartitionedCall-10632069С
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall*conv2d_25/StatefulPartitionedCall:output:05batch_normalization_20_statefulpartitionedcall_args_15batch_normalization_20_statefulpartitionedcall_args_25batch_normalization_20_statefulpartitionedcall_args_35batch_normalization_20_statefulpartitionedcall_args_4*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin	
2*/
_gradient_op_typePartitionedCall-10632826*]
fXRV
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10632813*
Tout
2ы
leaky_re_lu_20/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10632850*U
fPRN
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10632844*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         H@*
Tin
2њ
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_20/PartitionedCall:output:0(conv2d_26_statefulpartitionedcall_args_1*/
_output_shapes
:         F@*
Tin
2*/
_gradient_op_typePartitionedCall-10632231*P
fKRI
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8С
.batch_normalization_21/StatefulPartitionedCallStatefulPartitionedCall*conv2d_26/StatefulPartitionedCall:output:05batch_normalization_21_statefulpartitionedcall_args_15batch_normalization_21_statefulpartitionedcall_args_25batch_normalization_21_statefulpartitionedcall_args_35batch_normalization_21_statefulpartitionedcall_args_4*/
_output_shapes
:         F@*
Tin	
2*/
_gradient_op_typePartitionedCall-10632929*]
fXRV
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10632916*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8ы
leaky_re_lu_21/PartitionedCallPartitionedCall7batch_normalization_21/StatefulPartitionedCall:output:0*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         F@*/
_gradient_op_typePartitionedCall-10632953*U
fPRN
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10632947*
Tout
2њ
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_21/PartitionedCall:output:0(conv2d_27_statefulpartitionedcall_args_1*/
_gradient_op_typePartitionedCall-10632393*P
fKRI
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         D@*
Tin
2С
.batch_normalization_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:05batch_normalization_22_statefulpartitionedcall_args_15batch_normalization_22_statefulpartitionedcall_args_25batch_normalization_22_statefulpartitionedcall_args_35batch_normalization_22_statefulpartitionedcall_args_4*/
_gradient_op_typePartitionedCall-10633032*]
fXRV
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10633019*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*/
_output_shapes
:         D@ы
leaky_re_lu_22/PartitionedCallPartitionedCall7batch_normalization_22/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10633056*U
fPRN
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10633050*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         D@њ
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_22/PartitionedCall:output:0(conv2d_28_statefulpartitionedcall_args_1*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         B@*
Tin
2*/
_gradient_op_typePartitionedCall-10632555*P
fKRI
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549*
Tout
2С
.batch_normalization_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:05batch_normalization_23_statefulpartitionedcall_args_15batch_normalization_23_statefulpartitionedcall_args_25batch_normalization_23_statefulpartitionedcall_args_35batch_normalization_23_statefulpartitionedcall_args_4*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10633122*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         B@*
Tin	
2*/
_gradient_op_typePartitionedCall-10633135ы
leaky_re_lu_23/PartitionedCallPartitionedCall7batch_normalization_23/StatefulPartitionedCall:output:0*/
_gradient_op_typePartitionedCall-10633159*U
fPRN
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10633153*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*/
_output_shapes
:         B@*
Tin
2й
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall'leaky_re_lu_23/PartitionedCall:output:0(conv2d_29_statefulpartitionedcall_args_1(conv2d_29_statefulpartitionedcall_args_2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10632720*P
fKRI
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714*
Tout
2┌
softmax_2/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*/
_output_shapes
:         @	*
Tin
2*/
_gradient_op_typePartitionedCall-10633180*P
fKRI
G__inference_softmax_2_layer_call_and_return_conditional_losses_10633174*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8н
add_3/PartitionedCallPartitionedCall"softmax_2/PartitionedCall:output:0input_8*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin
2*/
_output_shapes
:         @	*/
_gradient_op_typePartitionedCall-10633200*L
fGRE
C__inference_add_3_layer_call_and_return_conditional_losses_10633193Т
IdentityIdentityadd_3/PartitionedCall:output:0/^batch_normalization_20/StatefulPartitionedCall/^batch_normalization_21/StatefulPartitionedCall/^batch_normalization_22/StatefulPartitionedCall/^batch_normalization_23/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*
T0*/
_output_shapes
:         @	"
identityIdentity:output:0*Б
_input_shapesЉ
ј:         @	:         @	::::::::::::::::::::::2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2`
.batch_normalization_21/StatefulPartitionedCall.batch_normalization_21/StatefulPartitionedCall2`
.batch_normalization_22/StatefulPartitionedCall.batch_normalization_22/StatefulPartitionedCall2`
.batch_normalization_23/StatefulPartitionedCall.batch_normalization_23/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall:' #
!
_user_specified_name	input_7:'#
!
_user_specified_name	input_8: : : : : : : :	 :
 : : : : : : : : : : : : : 
в
ѓ
9__inference_batch_normalization_23_layer_call_fn_10634524

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityѕбStatefulPartitionedCall▀
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tout
2*/
config_proto

CPU

GPU2*0,1J 8*
Tin	
2*A
_output_shapes/
-:+                           @*/
_gradient_op_typePartitionedCall-10632696*]
fXRV
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10632695ю
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+                           @"
identityIdentity:output:0*P
_input_shapes?
=:+                           @::::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: : : 
Џ
h
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10634353

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         D@g
IdentityIdentityLeakyRelu:activations:0*/
_output_shapes
:         D@*
T0"
identityIdentity:output:0*.
_input_shapes
:         D@:& "
 
_user_specified_nameinputs
Ш
j
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026

inputs
identity}
Pad/paddingsConst*
_output_shapes

:*9
value0B."                             *
dtype0~
PadPadinputsPad/paddings:output:0*J
_output_shapes8
6:4                                    *
T0w
IdentityIdentityPad:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Ш
j
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044

inputs
identity}
Pad/paddingsConst*9
value0B."                             *
dtype0*
_output_shapes

:~
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4                                    w
IdentityIdentityPad:output:0*J
_output_shapes8
6:4                                    *
T0"
identityIdentity:output:0*I
_input_shapes8
6:4                                    :& "
 
_user_specified_nameinputs
Џ
h
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10634529

inputs
identityO
	LeakyRelu	LeakyReluinputs*/
_output_shapes
:         B@g
IdentityIdentityLeakyRelu:activations:0*
T0*/
_output_shapes
:         B@"
identityIdentity:output:0*.
_input_shapes
:         B@:& "
 
_user_specified_nameinputs"wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*§
serving_defaultж
C
input_78
serving_default_input_7:0         @	
C
input_88
serving_default_input_8:0         @	A
add_38
StatefulPartitionedCall:0         @	tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:ћ╝
ъІ
layer-0
layer-1
layer-2
layer-3
layer-4
layer_with_weights-0
layer-5
layer_with_weights-1
layer-6
layer-7
	layer_with_weights-2
	layer-8

layer_with_weights-3

layer-9
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer-13
layer_with_weights-6
layer-14
layer_with_weights-7
layer-15
layer-16
layer_with_weights-8
layer-17
layer-18
layer-19
regularization_losses
trainable_variables
	variables
	keras_api

signatures
п__call__
┘_default_save_signature
+┌&call_and_return_all_conditional_losses"ТЁ
_tf_keras_model╦Ё{"class_name": "Model", "name": "generator_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "generator_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 16, 64, 9], "dtype": "float32", "sparse": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 16, 64, 9], "dtype": "float32", "sparse": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_6", "trainable": true, "dtype": "float32", "padding": [[5, 5], [5, 5]], "data_format": "channels_last"}, "name": "zero_padding2d_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_7", "trainable": true, "dtype": "float32", "padding": [[5, 5], [5, 5]], "data_format": "channels_last"}, "name": "zero_padding2d_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["zero_padding2d_6", 0, 0, {}], ["zero_padding2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_20", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_20", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["leaky_re_lu_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_21", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_21", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["leaky_re_lu_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_22", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_22", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["leaky_re_lu_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_23", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["leaky_re_lu_23", 0, 0, {}]]]}, {"class_name": "Softmax", "config": {"name": "softmax_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "softmax_2", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["softmax_2", 0, 0, {}], ["input_8", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["add_3", 0, 0]]}, "input_spec": [null, null], "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "generator_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 16, 64, 9], "dtype": "float32", "sparse": false, "name": "input_7"}, "name": "input_7", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": [null, 16, 64, 9], "dtype": "float32", "sparse": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_6", "trainable": true, "dtype": "float32", "padding": [[5, 5], [5, 5]], "data_format": "channels_last"}, "name": "zero_padding2d_6", "inbound_nodes": [[["input_7", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_7", "trainable": true, "dtype": "float32", "padding": [[5, 5], [5, 5]], "data_format": "channels_last"}, "name": "zero_padding2d_7", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["zero_padding2d_6", 0, 0, {}], ["zero_padding2d_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_20", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_20", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_20", "inbound_nodes": [[["batch_normalization_20", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_26", "inbound_nodes": [[["leaky_re_lu_20", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_21", "inbound_nodes": [[["conv2d_26", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_21", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_21", "inbound_nodes": [[["batch_normalization_21", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["leaky_re_lu_21", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_22", "inbound_nodes": [[["conv2d_27", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_22", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_22", "inbound_nodes": [[["batch_normalization_22", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_28", "inbound_nodes": [[["leaky_re_lu_22", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_23", "inbound_nodes": [[["conv2d_28", 0, 0, {}]]]}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_23", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}, "name": "leaky_re_lu_23", "inbound_nodes": [[["batch_normalization_23", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_29", "inbound_nodes": [[["leaky_re_lu_23", 0, 0, {}]]]}, {"class_name": "Softmax", "config": {"name": "softmax_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "softmax_2", "inbound_nodes": [[["conv2d_29", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["softmax_2", 0, 0, {}], ["input_8", 0, 0, {}]]]}], "input_layers": [["input_7", 0, 0], ["input_8", 0, 0]], "output_layers": [["add_3", 0, 0]]}}}
│
regularization_losses
trainable_variables
	variables
	keras_api
█__call__
+▄&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"class_name": "InputLayer", "name": "input_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 16, 64, 9], "config": {"batch_input_shape": [null, 16, 64, 9], "dtype": "float32", "sparse": false, "name": "input_7"}}
│
regularization_losses
trainable_variables
 	variables
!	keras_api
П__call__
+я&call_and_return_all_conditional_losses"б
_tf_keras_layerѕ{"class_name": "InputLayer", "name": "input_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 16, 64, 9], "config": {"batch_input_shape": [null, 16, 64, 9], "dtype": "float32", "sparse": false, "name": "input_8"}}
с
"regularization_losses
#trainable_variables
$	variables
%	keras_api
▀__call__
+Я&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "ZeroPadding2D", "name": "zero_padding2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "zero_padding2d_6", "trainable": true, "dtype": "float32", "padding": [[5, 5], [5, 5]], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
с
&regularization_losses
'trainable_variables
(	variables
)	keras_api
р__call__
+Р&call_and_return_all_conditional_losses"м
_tf_keras_layerИ{"class_name": "ZeroPadding2D", "name": "zero_padding2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "zero_padding2d_7", "trainable": true, "dtype": "float32", "padding": [[5, 5], [5, 5]], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Ш
*regularization_losses
+trainable_variables
,	variables
-	keras_api
с__call__
+С&call_and_return_all_conditional_losses"т
_tf_keras_layer╦{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}}
Ё

.kernel
/regularization_losses
0trainable_variables
1	variables
2	keras_api
т__call__
+Т&call_and_return_all_conditional_losses"У
_tf_keras_layer╬{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 9}}}}
У
3axis
	4gamma
5beta
6moving_mean
7moving_variance
8regularization_losses
9trainable_variables
:	variables
;	keras_api
у__call__
+У&call_and_return_all_conditional_losses"њ
_tf_keras_layerЭ{"class_name": "BatchNormalization", "name": "batch_normalization_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_20", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
г
<regularization_losses
=trainable_variables
>	variables
?	keras_api
ж__call__
+Ж&call_and_return_all_conditional_losses"Џ
_tf_keras_layerЂ{"class_name": "LeakyReLU", "name": "leaky_re_lu_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_20", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
є

@kernel
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
в__call__
+В&call_and_return_all_conditional_losses"ж
_tf_keras_layer¤{"class_name": "Conv2D", "name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
У
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses"њ
_tf_keras_layerЭ{"class_name": "BatchNormalization", "name": "batch_normalization_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_21", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
г
Nregularization_losses
Otrainable_variables
P	variables
Q	keras_api
№__call__
+­&call_and_return_all_conditional_losses"Џ
_tf_keras_layerЂ{"class_name": "LeakyReLU", "name": "leaky_re_lu_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_21", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
є

Rkernel
Sregularization_losses
Ttrainable_variables
U	variables
V	keras_api
ы__call__
+Ы&call_and_return_all_conditional_losses"ж
_tf_keras_layer¤{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
У
Waxis
	Xgamma
Ybeta
Zmoving_mean
[moving_variance
\regularization_losses
]trainable_variables
^	variables
_	keras_api
з__call__
+З&call_and_return_all_conditional_losses"њ
_tf_keras_layerЭ{"class_name": "BatchNormalization", "name": "batch_normalization_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_22", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
г
`regularization_losses
atrainable_variables
b	variables
c	keras_api
ш__call__
+Ш&call_and_return_all_conditional_losses"Џ
_tf_keras_layerЂ{"class_name": "LeakyReLU", "name": "leaky_re_lu_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_22", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
є

dkernel
eregularization_losses
ftrainable_variables
g	variables
h	keras_api
э__call__
+Э&call_and_return_all_conditional_losses"ж
_tf_keras_layer¤{"class_name": "Conv2D", "name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
У
iaxis
	jgamma
kbeta
lmoving_mean
mmoving_variance
nregularization_losses
otrainable_variables
p	variables
q	keras_api
щ__call__
+Щ&call_and_return_all_conditional_losses"њ
_tf_keras_layerЭ{"class_name": "BatchNormalization", "name": "batch_normalization_23", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_23", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "RandomNormal", "config": {"mean": 1.0, "stddev": 0.02, "seed": null}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
г
rregularization_losses
strainable_variables
t	variables
u	keras_api
ч__call__
+Ч&call_and_return_all_conditional_losses"Џ
_tf_keras_layerЂ{"class_name": "LeakyReLU", "name": "leaky_re_lu_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "leaky_re_lu_23", "trainable": true, "dtype": "float32", "alpha": 0.20000000298023224}}
ј

vkernel
wbias
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
§__call__
+■&call_and_return_all_conditional_losses"у
_tf_keras_layer═{"class_name": "Conv2D", "name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 9, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
ј
|regularization_losses
}trainable_variables
~	variables
	keras_api
 __call__
+ђ&call_and_return_all_conditional_losses"§
_tf_keras_layerс{"class_name": "Softmax", "name": "softmax_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "softmax_2", "trainable": true, "dtype": "float32", "axis": -1}}
Щ
ђregularization_losses
Ђtrainable_variables
ѓ	variables
Ѓ	keras_api
Ђ__call__
+ѓ&call_and_return_all_conditional_losses"т
_tf_keras_layer╦{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}}
 "
trackable_list_wrapper
є
.0
41
52
@3
F4
G5
R6
X7
Y8
d9
j10
k11
v12
w13"
trackable_list_wrapper
к
.0
41
52
63
74
@5
F6
G7
H8
I9
R10
X11
Y12
Z13
[14
d15
j16
k17
l18
m19
v20
w21"
trackable_list_wrapper
┐
regularization_losses
ёnon_trainable_variables
Ёlayers
єmetrics
trainable_variables
 Єlayer_regularization_losses
	variables
п__call__
┘_default_save_signature
+┌&call_and_return_all_conditional_losses
'┌"call_and_return_conditional_losses"
_generic_user_object
-
Ѓserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
regularization_losses
ѕnon_trainable_variables
Ѕlayers
іmetrics
trainable_variables
 Іlayer_regularization_losses
	variables
█__call__
+▄&call_and_return_all_conditional_losses
'▄"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
regularization_losses
їnon_trainable_variables
Їlayers
јmetrics
trainable_variables
 Јlayer_regularization_losses
 	variables
П__call__
+я&call_and_return_all_conditional_losses
'я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
"regularization_losses
љnon_trainable_variables
Љlayers
њmetrics
#trainable_variables
 Њlayer_regularization_losses
$	variables
▀__call__
+Я&call_and_return_all_conditional_losses
'Я"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
&regularization_losses
ћnon_trainable_variables
Ћlayers
ќmetrics
'trainable_variables
 Ќlayer_regularization_losses
(	variables
р__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
*regularization_losses
ўnon_trainable_variables
Ўlayers
џmetrics
+trainable_variables
 Џlayer_regularization_losses
,	variables
с__call__
+С&call_and_return_all_conditional_losses
'С"call_and_return_conditional_losses"
_generic_user_object
*:(	@2conv2d_25/kernel
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
'
.0"
trackable_list_wrapper
А
/regularization_losses
юnon_trainable_variables
Юlayers
ъmetrics
0trainable_variables
 Ъlayer_regularization_losses
1	variables
т__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_20/gamma
):'@2batch_normalization_20/beta
2:0@ (2"batch_normalization_20/moving_mean
6:4@ (2&batch_normalization_20/moving_variance
 "
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
<
40
51
62
73"
trackable_list_wrapper
А
8regularization_losses
аnon_trainable_variables
Аlayers
бmetrics
9trainable_variables
 Бlayer_regularization_losses
:	variables
у__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
<regularization_losses
цnon_trainable_variables
Цlayers
дmetrics
=trainable_variables
 Дlayer_regularization_losses
>	variables
ж__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_26/kernel
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
'
@0"
trackable_list_wrapper
А
Aregularization_losses
еnon_trainable_variables
Еlayers
фmetrics
Btrainable_variables
 Фlayer_regularization_losses
C	variables
в__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_21/gamma
):'@2batch_normalization_21/beta
2:0@ (2"batch_normalization_21/moving_mean
6:4@ (2&batch_normalization_21/moving_variance
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
А
Jregularization_losses
гnon_trainable_variables
Гlayers
«metrics
Ktrainable_variables
 »layer_regularization_losses
L	variables
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
Nregularization_losses
░non_trainable_variables
▒layers
▓metrics
Otrainable_variables
 │layer_regularization_losses
P	variables
№__call__
+­&call_and_return_all_conditional_losses
'­"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_27/kernel
 "
trackable_list_wrapper
'
R0"
trackable_list_wrapper
'
R0"
trackable_list_wrapper
А
Sregularization_losses
┤non_trainable_variables
хlayers
Хmetrics
Ttrainable_variables
 иlayer_regularization_losses
U	variables
ы__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_22/gamma
):'@2batch_normalization_22/beta
2:0@ (2"batch_normalization_22/moving_mean
6:4@ (2&batch_normalization_22/moving_variance
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
<
X0
Y1
Z2
[3"
trackable_list_wrapper
А
\regularization_losses
Иnon_trainable_variables
╣layers
║metrics
]trainable_variables
 ╗layer_regularization_losses
^	variables
з__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
`regularization_losses
╝non_trainable_variables
йlayers
Йmetrics
atrainable_variables
 ┐layer_regularization_losses
b	variables
ш__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_28/kernel
 "
trackable_list_wrapper
'
d0"
trackable_list_wrapper
'
d0"
trackable_list_wrapper
А
eregularization_losses
└non_trainable_variables
┴layers
┬metrics
ftrainable_variables
 ├layer_regularization_losses
g	variables
э__call__
+Э&call_and_return_all_conditional_losses
'Э"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(@2batch_normalization_23/gamma
):'@2batch_normalization_23/beta
2:0@ (2"batch_normalization_23/moving_mean
6:4@ (2&batch_normalization_23/moving_variance
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
<
j0
k1
l2
m3"
trackable_list_wrapper
А
nregularization_losses
─non_trainable_variables
┼layers
кmetrics
otrainable_variables
 Кlayer_regularization_losses
p	variables
щ__call__
+Щ&call_and_return_all_conditional_losses
'Щ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
rregularization_losses
╚non_trainable_variables
╔layers
╩metrics
strainable_variables
 ╦layer_regularization_losses
t	variables
ч__call__
+Ч&call_and_return_all_conditional_losses
'Ч"call_and_return_conditional_losses"
_generic_user_object
*:(@	2conv2d_29/kernel
:	2conv2d_29/bias
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
А
xregularization_losses
╠non_trainable_variables
═layers
╬metrics
ytrainable_variables
 ¤layer_regularization_losses
z	variables
§__call__
+■&call_and_return_all_conditional_losses
'■"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
А
|regularization_losses
лnon_trainable_variables
Лlayers
мmetrics
}trainable_variables
 Мlayer_regularization_losses
~	variables
 __call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ц
ђregularization_losses
нnon_trainable_variables
Нlayers
оmetrics
Ђtrainable_variables
 Оlayer_regularization_losses
ѓ	variables
Ђ__call__
+ѓ&call_and_return_all_conditional_losses
'ѓ"call_and_return_conditional_losses"
_generic_user_object
X
60
71
H2
I3
Z4
[5
l6
m7"
trackable_list_wrapper
Х
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
є2Ѓ
.__inference_generator_2_layer_call_fn_10633330
.__inference_generator_2_layer_call_fn_10633818
.__inference_generator_2_layer_call_fn_10633406
.__inference_generator_2_layer_call_fn_10633790└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ў2ќ
#__inference__wrapped_model_10632017Ь
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *^б[
YџV
)і&
input_7         @	
)і&
input_8         @	
Ы2№
I__inference_generator_2_layer_call_and_return_conditional_losses_10633656
I__inference_generator_2_layer_call_and_return_conditional_losses_10633208
I__inference_generator_2_layer_call_and_return_conditional_losses_10633762
I__inference_generator_2_layer_call_and_return_conditional_losses_10633255└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
Џ2ў
3__inference_zero_padding2d_6_layer_call_fn_10632035Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Х2│
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Џ2ў
3__inference_zero_padding2d_7_layer_call_fn_10632053Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
Х2│
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044Я
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *@б=
;і84                                    
м2¤
(__inference_add_2_layer_call_fn_10633830б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_add_2_layer_call_and_return_conditional_losses_10633824б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
І2ѕ
,__inference_conv2d_25_layer_call_fn_10632073О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           	
д2Б
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           	
д2Б
9__inference_batch_normalization_20_layer_call_fn_10633987
9__inference_batch_normalization_20_layer_call_fn_10633911
9__inference_batch_normalization_20_layer_call_fn_10633920
9__inference_batch_normalization_20_layer_call_fn_10633996┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633902
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633880
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633956
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633978┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
█2п
1__inference_leaky_re_lu_20_layer_call_fn_10634006б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10634001б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
І2ѕ
,__inference_conv2d_26_layer_call_fn_10632235О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
9__inference_batch_normalization_21_layer_call_fn_10634087
9__inference_batch_normalization_21_layer_call_fn_10634172
9__inference_batch_normalization_21_layer_call_fn_10634096
9__inference_batch_normalization_21_layer_call_fn_10634163┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634056
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634132
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634078
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634154┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
█2п
1__inference_leaky_re_lu_21_layer_call_fn_10634182б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10634177б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
І2ѕ
,__inference_conv2d_27_layer_call_fn_10632397О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
9__inference_batch_normalization_22_layer_call_fn_10634263
9__inference_batch_normalization_22_layer_call_fn_10634272
9__inference_batch_normalization_22_layer_call_fn_10634348
9__inference_batch_normalization_22_layer_call_fn_10634339┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634308
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634254
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634330
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634232┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
█2п
1__inference_leaky_re_lu_22_layer_call_fn_10634358б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10634353б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
І2ѕ
,__inference_conv2d_28_layer_call_fn_10632559О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
9__inference_batch_normalization_23_layer_call_fn_10634439
9__inference_batch_normalization_23_layer_call_fn_10634515
9__inference_batch_normalization_23_layer_call_fn_10634448
9__inference_batch_normalization_23_layer_call_fn_10634524┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
њ2Ј
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634506
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634484
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634430
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634408┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
█2п
1__inference_leaky_re_lu_23_layer_call_fn_10634534б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10634529б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
І2ѕ
,__inference_conv2d_29_layer_call_fn_10632725О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
д2Б
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714О
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *7б4
2і/+                           @
о2М
,__inference_softmax_2_layer_call_fn_10634544б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ы2Ь
G__inference_softmax_2_layer_call_and_return_conditional_losses_10634539б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_add_3_layer_call_fn_10634556б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_add_3_layer_call_and_return_conditional_losses_10634550б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
<B:
&__inference_signature_wrapper_10633492input_7input_8К
9__inference_batch_normalization_20_layer_call_fn_10633911Ѕ4567MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @ы
N__inference_zero_padding2d_6_layer_call_and_return_conditional_losses_10632026ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ љ
1__inference_leaky_re_lu_23_layer_call_fn_10634534[7б4
-б*
(і%
inputs         B@
ф " і         B@№
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633880ќ4567MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ █
G__inference_conv2d_25_layer_call_and_return_conditional_losses_10632063Ј.IбF
?б<
:і7
inputs+                           	
ф "?б<
5і2
0+                           @
џ К
9__inference_batch_normalization_20_layer_call_fn_10633920Ѕ4567MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @╩
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634232rXYZ[;б8
1б.
(і%
inputs         D@
p
ф "-б*
#і 
0         D@
џ ╩
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634408rjklm;б8
1б.
(і%
inputs         B@
p
ф "-б*
#і 
0         B@
џ №
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634056ќFGHIMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ █
G__inference_conv2d_28_layer_call_and_return_conditional_losses_10632549ЈdIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           @
џ ╗
(__inference_add_2_layer_call_fn_10633830јjбg
`б]
[џX
*і'
inputs/0         J	
*і'
inputs/1         J	
ф " і         J	│
G__inference_softmax_2_layer_call_and_return_conditional_losses_10634539h7б4
-б*
(і%
inputs         @	
ф "-б*
#і 
0         @	
џ ╩
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634430rjklm;б8
1б.
(і%
inputs         B@
p 
ф "-б*
#і 
0         B@
џ █
G__inference_conv2d_27_layer_call_and_return_conditional_losses_10632387ЈRIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           @
џ ╩
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634254rXYZ[;б8
1б.
(і%
inputs         D@
p 
ф "-б*
#і 
0         D@
џ ╩
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634132rFGHI;б8
1б.
(і%
inputs         F@
p
ф "-б*
#і 
0         F@
џ ╩
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633956r4567;б8
1б.
(і%
inputs         H@
p
ф "-б*
#і 
0         H@
џ №
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634078ќFGHIMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ р
#__inference__wrapped_model_10632017╣.4567@FGHIRXYZ[djklmvwhбe
^б[
YџV
)і&
input_7         @	
)і&
input_8         @	
ф "5ф2
0
add_3'і$
add_3         @	№
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634308ќXYZ[MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ №
T__inference_batch_normalization_22_layer_call_and_return_conditional_losses_10634330ќXYZ[MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ №
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634506ќjklmMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ ╩
T__inference_batch_normalization_21_layer_call_and_return_conditional_losses_10634154rFGHI;б8
1б.
(і%
inputs         F@
p 
ф "-б*
#і 
0         F@
џ ╩
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633978r4567;б8
1б.
(і%
inputs         H@
p 
ф "-б*
#і 
0         H@
џ б
9__inference_batch_normalization_20_layer_call_fn_10633987e4567;б8
1б.
(і%
inputs         H@
p
ф " і         H@№
T__inference_batch_normalization_23_layer_call_and_return_conditional_losses_10634484ќjklmMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ б
9__inference_batch_normalization_20_layer_call_fn_10633996e4567;б8
1б.
(і%
inputs         H@
p 
ф " і         H@љ
1__inference_leaky_re_lu_20_layer_call_fn_10634006[7б4
-б*
(і%
inputs         H@
ф " і         H@╔
3__inference_zero_padding2d_7_layer_call_fn_10632053ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    К
9__inference_batch_normalization_21_layer_call_fn_10634087ЅFGHIMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @│
,__inference_conv2d_25_layer_call_fn_10632073ѓ.IбF
?б<
:і7
inputs+                           	
ф "2і/+                           @К
9__inference_batch_normalization_21_layer_call_fn_10634096ЅFGHIMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @╗
(__inference_add_3_layer_call_fn_10634556јjбg
`б]
[џX
*і'
inputs/0         @	
*і'
inputs/1         @	
ф " і         @	╔
3__inference_zero_padding2d_6_layer_call_fn_10632035ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    б
9__inference_batch_normalization_21_layer_call_fn_10634163eFGHI;б8
1б.
(і%
inputs         F@
p
ф " і         F@с
C__inference_add_2_layer_call_and_return_conditional_losses_10633824Џjбg
`б]
[џX
*і'
inputs/0         J	
*і'
inputs/1         J	
ф "-б*
#і 
0         J	
џ │
,__inference_conv2d_26_layer_call_fn_10632235ѓ@IбF
?б<
:і7
inputs+                           @
ф "2і/+                           @б
9__inference_batch_normalization_21_layer_call_fn_10634172eFGHI;б8
1б.
(і%
inputs         F@
p 
ф " і         F@б
9__inference_batch_normalization_22_layer_call_fn_10634263eXYZ[;б8
1б.
(і%
inputs         D@
p
ф " і         D@б
9__inference_batch_normalization_22_layer_call_fn_10634272eXYZ[;б8
1б.
(і%
inputs         D@
p 
ф " і         D@І
,__inference_softmax_2_layer_call_fn_10634544[7б4
-б*
(і%
inputs         @	
ф " і         @	К
9__inference_batch_normalization_22_layer_call_fn_10634339ЅXYZ[MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @с
C__inference_add_3_layer_call_and_return_conditional_losses_10634550Џjбg
`б]
[џX
*і'
inputs/0         @	
*і'
inputs/1         @	
ф "-б*
#і 
0         @	
џ И
L__inference_leaky_re_lu_20_layer_call_and_return_conditional_losses_10634001h7б4
-б*
(і%
inputs         H@
ф "-б*
#і 
0         H@
џ К
9__inference_batch_normalization_22_layer_call_fn_10634348ЅXYZ[MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @б
9__inference_batch_normalization_23_layer_call_fn_10634439ejklm;б8
1б.
(і%
inputs         B@
p
ф " і         B@Ѕ
I__inference_generator_2_layer_call_and_return_conditional_losses_10633656╗.4567@FGHIRXYZ[djklmvwrбo
hбe
[џX
*і'
inputs/0         @	
*і'
inputs/1         @	
p

 
ф "-б*
#і 
0         @	
џ љ
1__inference_leaky_re_lu_21_layer_call_fn_10634182[7б4
-б*
(і%
inputs         F@
ф " і         F@б
9__inference_batch_normalization_23_layer_call_fn_10634448ejklm;б8
1б.
(і%
inputs         B@
p 
ф " і         B@р
.__inference_generator_2_layer_call_fn_10633818«.4567@FGHIRXYZ[djklmvwrбo
hбe
[џX
*і'
inputs/0         @	
*і'
inputs/1         @	
p 

 
ф " і         @	ш
&__inference_signature_wrapper_10633492╩.4567@FGHIRXYZ[djklmvwyбv
б 
oфl
4
input_7)і&
input_7         @	
4
input_8)і&
input_8         @	"5ф2
0
add_3'і$
add_3         @	Є
I__inference_generator_2_layer_call_and_return_conditional_losses_10633208╣.4567@FGHIRXYZ[djklmvwpбm
fбc
YџV
)і&
input_7         @	
)і&
input_8         @	
p

 
ф "-б*
#і 
0         @	
џ ы
N__inference_zero_padding2d_7_layer_call_and_return_conditional_losses_10632044ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ▀
.__inference_generator_2_layer_call_fn_10633330г.4567@FGHIRXYZ[djklmvwpбm
fбc
YџV
)і&
input_7         @	
)і&
input_8         @	
p

 
ф " і         @	К
9__inference_batch_normalization_23_layer_call_fn_10634515ЅjklmMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @р
.__inference_generator_2_layer_call_fn_10633790«.4567@FGHIRXYZ[djklmvwrбo
hбe
[џX
*і'
inputs/0         @	
*і'
inputs/1         @	
p

 
ф " і         @	│
,__inference_conv2d_27_layer_call_fn_10632397ѓRIбF
?б<
:і7
inputs+                           @
ф "2і/+                           @К
9__inference_batch_normalization_23_layer_call_fn_10634524ЅjklmMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @И
L__inference_leaky_re_lu_22_layer_call_and_return_conditional_losses_10634353h7б4
-б*
(і%
inputs         D@
ф "-б*
#і 
0         D@
џ И
L__inference_leaky_re_lu_23_layer_call_and_return_conditional_losses_10634529h7б4
-б*
(і%
inputs         B@
ф "-б*
#і 
0         B@
џ И
L__inference_leaky_re_lu_21_layer_call_and_return_conditional_losses_10634177h7б4
-б*
(і%
inputs         F@
ф "-б*
#і 
0         F@
џ љ
1__inference_leaky_re_lu_22_layer_call_fn_10634358[7б4
-б*
(і%
inputs         D@
ф " і         D@▀
.__inference_generator_2_layer_call_fn_10633406г.4567@FGHIRXYZ[djklmvwpбm
fбc
YџV
)і&
input_7         @	
)і&
input_8         @	
p 

 
ф " і         @	│
,__inference_conv2d_28_layer_call_fn_10632559ѓdIбF
?б<
:і7
inputs+                           @
ф "2і/+                           @Ѕ
I__inference_generator_2_layer_call_and_return_conditional_losses_10633762╗.4567@FGHIRXYZ[djklmvwrбo
hбe
[џX
*і'
inputs/0         @	
*і'
inputs/1         @	
p 

 
ф "-б*
#і 
0         @	
џ №
T__inference_batch_normalization_20_layer_call_and_return_conditional_losses_10633902ќ4567MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ Є
I__inference_generator_2_layer_call_and_return_conditional_losses_10633255╣.4567@FGHIRXYZ[djklmvwpбm
fбc
YџV
)і&
input_7         @	
)і&
input_8         @	
p 

 
ф "-б*
#і 
0         @	
џ █
G__inference_conv2d_26_layer_call_and_return_conditional_losses_10632225Ј@IбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           @
џ ▄
G__inference_conv2d_29_layer_call_and_return_conditional_losses_10632714љvwIбF
?б<
:і7
inputs+                           @
ф "?б<
5і2
0+                           	
џ ┤
,__inference_conv2d_29_layer_call_fn_10632725ЃvwIбF
?б<
:і7
inputs+                           @
ф "2і/+                           	