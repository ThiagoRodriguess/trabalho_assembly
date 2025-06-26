# ============================================
# forward_pass: Executa inferência da rede
# ============================================
forward_pass:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    # Obtém número de camadas
    la t0, num_layers
    lw s0, 0(t0)
    
    # Processa camada 1: activation_0 -> activation_1
    la a0, activation_0
    la a1, weights_l1
    la a2, activation_1
    la t0, layer_sizes
    lw a3, 0(t0)
    lw a4, 4(t0)
    jal ra, matrix_multiply
    
    # Aplica ReLU
    la a0, activation_1
    la t0, layer_sizes
    lw a1, 4(t0)
    jal ra, apply_relu
    
    # Se só tem 2 valores, para aqui
    li t0, 2
    beq s0, t0, forward_done
    
    # Processa camada 2: activation_1 -> activation_2
    la a0, activation_1
    la a1, weights_l2
    la a2, activation_2
    la t0, layer_sizes
    lw a3, 4(t0)
    lw a4, 8(t0)
    jal ra, matrix_multiply
    
    # Aplica ReLU
    la a0, activation_2
    la t0, layer_sizes
    lw a1, 8(t0)
    jal ra, apply_relu
    
    # Se tem 3 valores, para aqui
    li t0, 3
    beq s0, t0, forward_done
    
    # Processa camada 3: activation_2 -> activation_3
    la a0, activation_2
    la a1, weights_l3
    la a2, activation_3
    la t0, layer_sizes
    lw a3, 8(t0)
    lw a4, 12(t0)
    jal ra, matrix_multiply
    
    # Se tem 4 valores, NÃO aplica ReLU (é a saída)
    li t0, 4
    beq s0, t0, forward_done
    
    # Aplica ReLU na camada 3
    la a0, activation_3
    la t0, layer_sizes
    lw a1, 12(t0)
    jal ra, apply_relu
    
    # Processa camada 4: activation_3 -> activation_4
    la a0, activation_3
    la a1, weights_l4
    la a2, activation_4
    la t0, layer_sizes
    lw a3, 12(t0)
    lw a4, 16(t0)
    jal ra, matrix_multiply
    
    # NÃO aplica ReLU na última camada
    
forward_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret
# find_argmax: Encontra índice do maior valor
# ============================================
# IrisNet - Implementação de Rede Neural em Assembly RISC-V
# Autor: Implementação para MC404
# Descrição: Inferência de rede neural para classificação do dataset Iris

.data
    # Buffers para armazenar dados
    architecture:   .space 64      # Buffer para arquitetura (ex: "4,10,20,3")
    json_buffer:    .space 8192    # Buffer para JSON com pesos
    input_buffer:   .space 32      # Buffer para entrada (4 valores)
    
    # Arrays para dados processados
    layer_sizes:    .space 20      # Até 5 camadas (4 bytes cada)
    num_layers:     .word 0        # Número de camadas
    
    # Matrizes de pesos (máximo 4 camadas de transição)
    weights_l1:     .space 2048    # Matriz de pesos camada 1
    weights_l2:     .space 2048    # Matriz de pesos camada 2
    weights_l3:     .space 2048    # Matriz de pesos camada 3
    weights_l4:     .space 2048    # Matriz de pesos camada 4
    
    # Vetores de ativação
    activation_0:   .space 128     # Entrada (máx 32 neurônios)
    activation_1:   .space 128     # Camada 1 (máx 32 neurônios)
    activation_2:   .space 128     # Camada 2 (máx 32 neurônios)
    activation_3:   .space 128     # Camada 3 (máx 32 neurônios)
    activation_4:   .space 128     # Camada 4 (máx 32 neurônios)
    
    # Strings auxiliares
    l1_str:         .string "\"l1\":"
    l2_str:         .string "\"l2\":"
    l3_str:         .string "\"l3\":"
    l4_str:         .string "\"l4\":"
    
    # Buffer para resultado
    result_buffer:  .space 4
    
.text
.globl _start

_start:
    # Lê entrada do usuário
    jal ra, read_input
    
    # Processa arquitetura da rede
    jal ra, parse_architecture
    
    # Processa JSON com pesos
    jal ra, parse_json
    
    # Processa entrada da rede
    jal ra, parse_input
    
    # Executa inferência
    jal ra, forward_pass
    
    # Encontra argmax e imprime resultado
    jal ra, find_argmax
    
    # Termina programa
    li a7, 93              # syscall exit
    li a0, 0               # exit code 0
    ecall

# ============================================
# read_input: Lê as 3 linhas de entrada (versão getchar)
# ============================================
read_input:
    addi sp, sp, -16
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    
    la s0, architecture     # Ponteiro para buffer atual
    li s1, 0               # Contador de newlines
    li s2, 0               # Índice no buffer atual
    
read_loop:
    # Cria buffer de 1 byte na stack
    addi sp, sp, -4
    
    # Lê um caractere
    li a0, 0               # stdin
    mv a1, sp              # buffer temporário
    li a2, 1               # 1 byte
    li a7, 63              # read syscall
    ecall
    
    # Verifica se leu algo
    beqz a0, end_read
    
    # Pega o caractere lido
    lb t0, 0(sp)
    addi sp, sp, 4
    
    # Verifica se é newline
    li t1, 10              # '\n' = 10
    beq t0, t1, handle_newline
    
    # Armazena caractere no buffer atual
    add t1, s0, s2
    sb t0, 0(t1)
    addi s2, s2, 1
    j read_loop
    
handle_newline:
    # Null terminate a linha
    add t1, s0, s2
    sb zero, 0(t1)
    
    # Incrementa contador de linhas
    addi s1, s1, 1
    
    # Verifica se já leu 3 linhas
    li t1, 3
    beq s1, t1, end_read
    
    # Muda para próximo buffer
    li t1, 1
    beq s1, t1, switch_json
    
    # Deve ser linha 2, muda para input
    la s0, input_buffer
    li s2, 0
    j read_loop
    
switch_json:
    la s0, json_buffer
    li s2, 0
    j read_loop
    
end_read:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    addi sp, sp, 16
    ret

# ============================================
# parse_architecture: Processa string da arquitetura
# Ex: "4,10,20,3" -> [4, 10, 20, 3]
# ============================================
parse_architecture:
    addi sp, sp, -16
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    
    la s0, architecture      # Ponteiro para string
    la s1, layer_sizes       # Ponteiro para array de tamanhos
    li s2, 0                 # Contador de camadas
    
parse_arch_loop:
    li t0, 0                 # Acumulador para número
    
parse_digit:
    lb t1, 0(s0)            # Lê caractere
    addi s0, s0, 1          # Avança ponteiro
    
    # Verifica se é dígito
    li t2, 48              # '0' = 48
    blt t1, t2, save_number
    li t2, 57              # '9' = 57
    bgt t1, t2, save_number
    
    # Converte dígito para número
    addi t1, t1, -48       # -'0'
    li t3, 10
    mul t0, t0, t3
    add t0, t0, t1
    j parse_digit
    
save_number:
    # Salva número no array
    sw t0, 0(s1)
    addi s1, s1, 4
    addi s2, s2, 1
    
    # Verifica se chegou ao fim
    li t2, 10              # '\n' = 10
    beq t1, t2, parse_arch_done
    li t2, 0               # '\0' = 0
    beq t1, t2, parse_arch_done
    
    j parse_arch_loop
    
parse_arch_done:
    # Salva número de camadas
    la t0, num_layers
    sw s2, 0(t0)
    
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    addi sp, sp, 16
    ret

# ============================================
# parse_json: Processa JSON com pesos
# ============================================
parse_json:
    addi sp, sp, -20
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    
    la s0, json_buffer       # Ponteiro para JSON
    
    # Processa camada 1
    la a0, l1_str
    la a1, weights_l1
    jal ra, parse_layer
    
    # Processa camada 2
    la a0, l2_str
    la a1, weights_l2
    jal ra, parse_layer
    
    # Processa camada 3
    la a0, l3_str
    la a1, weights_l3
    jal ra, parse_layer
    
    # Verifica se há camada 4
    la t0, num_layers
    lw t0, 0(t0)
    li t1, 5
    blt t0, t1, parse_json_done
    
    # Processa camada 4
    la a0, l4_str
    la a1, weights_l4
    jal ra, parse_layer
    
parse_json_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    addi sp, sp, 20
    ret

# ============================================
# parse_layer: Processa uma camada do JSON
# a0 = string da camada ("l1:", "l2:", etc)
# a1 = ponteiro para matriz de pesos
# ============================================
parse_layer:
    addi sp, sp, -24
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    
    mv s3, a0               # String da camada
    mv s4, a1               # Ponteiro para pesos
    la s0, json_buffer      # Ponteiro para JSON
    
    # Procura string da camada no JSON
find_layer:
    lb t0, 0(s0)
    lb t1, 0(s3)
    bne t0, t1, next_char
    
    # Verifica se encontrou a string completa
    mv t2, s0
    mv t3, s3
check_string:
    lb t0, 0(t2)
    lb t1, 0(t3)
    bne t0, t1, next_char
    addi t2, t2, 1
    addi t3, t3, 1
    lb t1, 0(t3)
    beqz t1, found_layer
    j check_string
    
next_char:
    addi s0, s0, 1
    j find_layer
    
found_layer:
    # Avança até o início da matriz [[
find_matrix_start:
    lb t0, 0(s0)
    li t1, 91              # '[' = 91
    bne t0, t1, skip_char
    lb t0, 1(s0)
    bne t0, t1, skip_char
    addi s0, s0, 2          # Pula [[
    j parse_matrix
skip_char:
    addi s0, s0, 1
    j find_matrix_start
    
parse_matrix:
    mv s1, s4               # Ponteiro atual para pesos
    
parse_neuron:
    # Verifica se é início de neurônio [
    lb t0, 0(s0)
    li t1, 91              # '[' = 91
    bne t0, t1, check_matrix_end
    addi s0, s0, 1          # Pula [
    
parse_weight:
    # Parse número (pode ser negativo)
    li t0, 0                # Acumulador
    li t2, 0                # Flag negativo
    
    # Verifica sinal negativo
    lb t1, 0(s0)
    li t3, 45              # '-' = 45
    bne t1, t3, parse_positive
    li t2, 1                # Marca como negativo
    addi s0, s0, 1          # Pula -
    
parse_positive:
    lb t1, 0(s0)
    li t3, 48              # '0' = 48
    blt t1, t3, end_number
    li t3, 57              # '9' = 57
    bgt t1, t3, end_number
    
    # Converte dígito
    addi t1, t1, -48       # -'0'
    li t3, 10
    mul t0, t0, t3
    add t0, t0, t1
    addi s0, s0, 1
    j parse_positive
    
end_number:
    # Aplica sinal se necessário
    beqz t2, store_weight
    sub t0, zero, t0
    
store_weight:
    # Armazena peso como byte
    sb t0, 0(s1)
    addi s1, s1, 1
    
    # Verifica próximo caractere
    lb t1, 0(s0)
    li t3, 44              # ',' = 44
    bne t1, t3, check_neuron_end
    addi s0, s0, 1          # Pula vírgula
    j parse_weight
    
check_neuron_end:
    li t3, 93              # ']' = 93
    bne t1, t3, parse_weight
    addi s0, s0, 1          # Pula ]
    
    # Verifica se há mais neurônios
    lb t1, 0(s0)
    li t3, 44              # ',' = 44
    bne t1, t3, check_matrix_end
    addi s0, s0, 1          # Pula vírgula
    j parse_neuron
    
check_matrix_end:
    lb t1, 0(s0)
    li t3, 93              # ']' = 93
    bne t1, t3, parse_neuron
    
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    addi sp, sp, 24
    ret

# ============================================
# parse_input: Processa entrada da rede
# ============================================
parse_input:
    addi sp, sp, -12
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    
    la s0, input_buffer      # Ponteiro para string
    la s1, activation_0      # Ponteiro para vetor de entrada
    li t4, 0                 # Contador de valores
    
parse_input_loop:
    li t0, 0                 # Acumulador
    
parse_input_digit:
    lb t1, 0(s0)
    addi s0, s0, 1
    
    # Verifica se é dígito
    li t2, 48              # '0' = 48
    blt t1, t2, save_input
    li t2, 57              # '9' = 57
    bgt t1, t2, save_input
    
    # Converte dígito
    addi t1, t1, -48       # -'0'
    li t3, 10
    mul t0, t0, t3
    add t0, t0, t1
    j parse_input_digit
    
save_input:
    # Salva valor como byte
    sb t0, 0(s1)
    addi s1, s1, 1
    addi t4, t4, 1
    
    # Verifica se leu 4 valores
    li t2, 4
    beq t4, t2, parse_input_done
    
    j parse_input_loop
    
parse_input_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    addi sp, sp, 12
    ret


# ============================================
# matrix_multiply: Multiplicação matriz-vetor
# a0 = vetor entrada
# a1 = matriz pesos
# a2 = vetor saída
# a3 = tamanho entrada
# a4 = tamanho saída
# ============================================
matrix_multiply:
    addi sp, sp, -24
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    sw s4, 20(sp)
    
    mv s0, a0               # Vetor entrada
    mv s1, a1               # Matriz pesos
    mv s2, a2               # Vetor saída
    mv s3, a3               # Tamanho entrada
    mv s4, a4               # Tamanho saída
    
    li t0, 0                # i = 0
outer_loop:
    bge t0, s4, mult_done   # i < tamanho_saída
    
    li t1, 0                # Acumulador
    li t2, 0                # j = 0
    
inner_loop:
    bge t2, s3, save_result # j < tamanho_entrada
    
    # Calcula índice na matriz: i * tamanho_entrada + j
    mul t3, t0, s3
    add t3, t3, t2
    add t3, s1, t3          # Endereço do peso
    
    # Carrega peso e entrada como bytes com sinal
    lb t4, 0(t3)            # weight[i][j]
    add t5, s0, t2          # Endereço entrada[j]
    lb t5, 0(t5)            # input[j]
    
    # Multiplica (já são valores com sinal correto devido ao lb)
    mul t6, t4, t5
    add t1, t1, t6
    
    addi t2, t2, 1          # j++
    j inner_loop
    
save_result:
    # Clamp para int8 (-128 a 127)
    li t3, 127
    bgt t1, t3, clamp_max
    li t3, -128
    blt t1, t3, clamp_min
    j store_result
    
clamp_max:
    li t1, 127
    j store_result
    
clamp_min:
    li t1, -128
    
store_result:
    # Salva resultado como byte
    add t3, s2, t0          # Endereço saída[i]
    sb t1, 0(t3)
    
    addi t0, t0, 1          # i++
    j outer_loop
    
mult_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    addi sp, sp, 24
    ret

# ============================================
# apply_relu: Aplica função ReLU
# a0 = vetor
# a1 = tamanho
# ============================================
apply_relu:
    addi sp, sp, -4
    sw ra, 0(sp)
    
    li t0, 0                # i = 0
relu_loop:
    bge t0, a1, relu_done   # i < tamanho
    
    add t1, a0, t0          # Endereço vetor[i]
    lb t2, 0(t1)            # Carrega valor (com sinal)
    
    # ReLU: max(0, x)
    bge t2, zero, skip_zero
    li t2, 0
    
skip_zero:
    sb t2, 0(t1)            # Salva resultado
    addi t0, t0, 1          # i++
    j relu_loop
    
relu_done:
    lw ra, 0(sp)
    addi sp, sp, 4
    ret

# ============================================
# find_argmax: Encontra índice do maior valor
# ============================================
find_argmax:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    # Determina qual vetor de ativação usar
    la t0, num_layers
    lw t0, 0(t0)
    
    # num_layers = 2: resultado em activation_1
    # num_layers = 3: resultado em activation_2
    # num_layers = 4: resultado em activation_3
    # num_layers = 5: resultado em activation_4
    
    li t1, 2
    beq t0, t1, use_act1
    li t1, 3
    beq t0, t1, use_act2
    li t1, 4
    beq t0, t1, use_act3
    
    # 5 camadas
    la s0, activation_4
    la t1, layer_sizes
    lw t2, 16(t1)           # Tamanho da última camada
    j do_argmax
    
use_act3:
    la s0, activation_3
    la t1, layer_sizes
    lw t2, 12(t1)
    j do_argmax
    
use_act2:
    la s0, activation_2
    la t1, layer_sizes
    lw t2, 8(t1)
    j do_argmax
    
use_act1:
    la s0, activation_1
    la t1, layer_sizes
    lw t2, 4(t1)
    
do_argmax:
    li t3, 0                # max_idx = 0
    lb t4, 0(s0)            # max_val = vec[0]
    
    li t0, 1                # i = 1
argmax_loop:
    bge t0, t2, print_result
    
    add t1, s0, t0          # Endereço vec[i]
    lb t5, 0(t1)            # vec[i]
    
    ble t5, t4, next_iter   # vec[i] <= max_val
    
    mv t3, t0               # max_idx = i
    mv t4, t5               # max_val = vec[i]
    
next_iter:
    addi t0, t0, 1          # i++
    j argmax_loop
    
print_result:
    # Converte índice para ASCII
    addi t3, t3, 48         # '0' = 48
    
    # Imprime resultado
    li a0, 1                # stdout
    la a1, result_buffer
    sb t3, 0(a1)
    li t4, 10               # '\n'
    sb t4, 1(a1)
    li a2, 2
    li a7, 64               # write
    ecall
    
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret
