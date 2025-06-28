# ============================================
# IrisNet - Real Neural Network with Matrix Multiplication
# ============================================

.data
    # Buffers de entrada
    architecture:   .space 64      
    json_buffer:    .space 2048    
    input_buffer:   .space 32      
    result_buffer:  .space 4       
    
    # Dados da rede neural
    layer_sizes:    .space 20      
    num_layers:     .word 0        
    
    # Vetores de ativação para cada camada
    activation_0:   .space 200     # Input layer (4 valores)
    activation_1:   .space 200     # L1 output (50 valores)
    activation_2:   .space 200     # L2 output (20 valores)  
    activation_3:   .space 200     # L3 output (10 valores)
    activation_4:   .space 200     # L4 output (3 valores)
    
    # Espaço para pesos das camadas
    weights_l1:     .space 800     # 4x50 = 200 weights (4 bytes each for indexing)
    weights_l2:     .space 4000    # 50x20 = 1000 weights  
    weights_l3:     .space 800     # 20x10 = 200 weights
    weights_l4:     .space 120     # 10x3 = 30 weights
    
    # Arquitetura da rede: 4 -> 50 -> 20 -> 10 -> 3
    arch_input:     .word 4
    arch_l1:        .word 50
    arch_l2:        .word 20
    arch_l3:        .word 10
    arch_l4:        .word 3

.text
.globl _start

_start:
    jal ra, read_input
    jal ra, parse_architecture_detailed
    jal ra, parse_json_weights
    jal ra, parse_input
    jal ra, forward_pass_real
    jal ra, find_argmax_output
    
    li a7, 93
    li a0, 0
    ecall

# ============================================
# read_input: Lê as 3 linhas
# ============================================
read_input:
    addi sp, sp, -16
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    
    la s0, architecture
    li s1, 0
    li s2, 0
    
read_loop:
    li a0, 0
    addi sp, sp, -4
    mv a1, sp
    li a2, 1
    li a7, 63
    ecall
    
    beqz a0, end_read
    
    lb t0, 0(sp)
    addi sp, sp, 4
    
    li t1, 10
    beq t0, t1, handle_newline
    
    li t1, 60
    bge s2, t1, read_loop
    
    add t1, s0, s2
    sb t0, 0(t1)
    addi s2, s2, 1
    j read_loop
    
handle_newline:
    add t1, s0, s2
    sb zero, 0(t1)
    
    addi s1, s1, 1
    li s2, 0
    
    li t1, 1
    beq s1, t1, switch_to_json
    li t1, 2
    beq s1, t1, switch_to_input
    j end_read
    
switch_to_json:
    la s0, json_buffer
    j read_loop
    
switch_to_input:
    la s0, input_buffer
    j read_loop
    
end_read:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    addi sp, sp, 16
    ret

# ============================================
# parse_architecture_detailed: Processa arquitetura
# ============================================
parse_architecture_detailed:
    addi sp, sp, -12
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    
    la s0, architecture
    la s1, layer_sizes
    li t2, 0
    li t3, 0
    
parse_arch_loop:
    lb t0, 0(s0)
    beqz t0, save_last_arch
    
    li t1, 48
    blt t0, t1, check_arch_delimiter
    li t1, 57
    bgt t0, t1, check_arch_delimiter
    
    addi t0, t0, -48
    mv t4, t3
    add t3, t4, t4
    add t3, t3, t3
    add t3, t3, t4
    add t3, t3, t3
    add t3, t3, t0
    j next_arch_char
    
check_arch_delimiter:
    li t1, 44
    beq t0, t1, save_arch_number
    li t1, 32
    beq t0, t1, save_arch_number
    j next_arch_char
    
save_arch_number:
    beqz t3, next_arch_char
    sw t3, 0(s1)
    addi s1, s1, 4
    addi t2, t2, 1
    li t3, 0
    
next_arch_char:
    addi s0, s0, 1
    j parse_arch_loop
    
save_last_arch:
    beqz t3, arch_setup_done
    sw t3, 0(s1)
    addi t2, t2, 1
    
arch_setup_done:
    la t0, num_layers
    sw t2, 0(t0)
    
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    addi sp, sp, 12
    ret

# ============================================
# parse_json_weights: Extrai todos os pesos do JSON
# ============================================
parse_json_weights:
    addi sp, sp, -16
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    
    # Parse L1 weights
    jal ra, find_l1_weights
    beqz a0, use_default_weights
    la s1, weights_l1
    jal ra, parse_matrix_weights
    
    # Parse L2 weights  
    jal ra, find_l2_weights
    beqz a0, use_default_weights
    la s1, weights_l2
    jal ra, parse_matrix_weights
    
    # Parse L3 weights
    jal ra, find_l3_weights
    beqz a0, use_default_weights
    la s1, weights_l3
    jal ra, parse_matrix_weights
    
    # Parse L4 weights
    jal ra, find_l4_weights
    beqz a0, use_default_weights
    la s1, weights_l4
    jal ra, parse_matrix_weights
    
    j json_weights_done
    
use_default_weights:
    jal ra, setup_default_weights
    
json_weights_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    addi sp, sp, 16
    ret

# ============================================
# find_l1_weights: Encontra "l1" no JSON
# ============================================
find_l1_weights:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    la s0, json_buffer
    
find_l1_loop:
    lb t0, 0(s0)
    beqz t0, find_l1_failed
    
    li t1, 108             # 'l'
    bne t0, t1, next_l1_char
    
    lb t1, 1(s0)
    li t2, 49              # '1'
    bne t1, t2, next_l1_char
    
    lb t1, 2(s0)
    li t2, 34              # '"'
    bne t1, t2, next_l1_char
    
    addi s0, s0, 3
    
find_l1_bracket:
    lb t0, 0(s0)
    beqz t0, find_l1_failed
    
    li t1, 91              # '['
    beq t0, t1, find_l1_success
    
    addi s0, s0, 1
    j find_l1_bracket
    
find_l1_success:
    addi s0, s0, 1
    li a0, 1
    j find_l1_end
    
find_l1_failed:
    li a0, 0
    
find_l1_end:
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret

next_l1_char:
    addi s0, s0, 1
    j find_l1_loop

# ============================================
# find_l2_weights, find_l3_weights, find_l4_weights
# ============================================
find_l2_weights:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    la s0, json_buffer
    
find_l2_loop:
    lb t0, 0(s0)
    beqz t0, find_l2_failed
    
    li t1, 108
    bne t0, t1, next_l2_char
    
    lb t1, 1(s0)
    li t2, 50              # '2'
    bne t1, t2, next_l2_char
    
    lb t1, 2(s0)
    li t2, 34
    bne t1, t2, next_l2_char
    
    addi s0, s0, 3
    
find_l2_bracket:
    lb t0, 0(s0)
    beqz t0, find_l2_failed
    
    li t1, 91
    beq t0, t1, find_l2_success
    
    addi s0, s0, 1
    j find_l2_bracket
    
find_l2_success:
    addi s0, s0, 1
    li a0, 1
    j find_l2_end
    
find_l2_failed:
    li a0, 0
    
find_l2_end:
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret

next_l2_char:
    addi s0, s0, 1
    j find_l2_loop

find_l3_weights:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    la s0, json_buffer
    
find_l3_loop:
    lb t0, 0(s0)
    beqz t0, find_l3_failed
    
    li t1, 108
    bne t0, t1, next_l3_char
    
    lb t1, 1(s0)
    li t2, 51              # '3'
    bne t1, t2, next_l3_char
    
    lb t1, 2(s0)
    li t2, 34
    bne t1, t2, next_l3_char
    
    addi s0, s0, 3
    
find_l3_bracket:
    lb t0, 0(s0)
    beqz t0, find_l3_failed
    
    li t1, 91
    beq t0, t1, find_l3_success
    
    addi s0, s0, 1
    j find_l3_bracket
    
find_l3_success:
    addi s0, s0, 1
    li a0, 1
    j find_l3_end
    
find_l3_failed:
    li a0, 0
    
find_l3_end:
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret

next_l3_char:
    addi s0, s0, 1
    j find_l3_loop

find_l4_weights:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    la s0, json_buffer
    
find_l4_loop:
    lb t0, 0(s0)
    beqz t0, find_l4_failed
    
    li t1, 108
    bne t0, t1, next_l4_char
    
    lb t1, 1(s0)
    li t2, 52              # '4'
    bne t1, t2, next_l4_char
    
    lb t1, 2(s0)
    li t2, 34
    bne t1, t2, next_l4_char
    
    addi s0, s0, 3
    
find_l4_bracket:
    lb t0, 0(s0)
    beqz t0, find_l4_failed
    
    li t1, 91
    beq t0, t1, find_l4_success
    
    addi s0, s0, 1
    j find_l4_bracket
    
find_l4_success:
    addi s0, s0, 1
    li a0, 1
    j find_l4_end
    
find_l4_failed:
    li a0, 0
    
find_l4_end:
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret

next_l4_char:
    addi s0, s0, 1
    j find_l4_loop

# ============================================
# parse_matrix_weights: Parse pesos para uma matriz
# ============================================
parse_matrix_weights:
    addi sp, sp, -16
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    
    # s0 = posição JSON, s1 = buffer destino
    li s2, 0                # contador
    li t3, 0                # número atual
    li t4, 0                # flag negativo
    
parse_matrix_loop:
    lb t0, 0(s0)
    beqz t0, matrix_parse_done
    
    li t1, 93               # ']'
    beq t0, t1, matrix_parse_done
    
    li t1, 45               # '-'
    beq t0, t1, set_matrix_negative
    
    li t1, 48
    blt t0, t1, check_matrix_delim
    li t1, 57
    bgt t0, t1, check_matrix_delim
    
    # Processa dígito
    addi t0, t0, -48
    mv t5, t3
    add t3, t5, t5
    add t3, t3, t3
    add t3, t3, t5
    add t3, t3, t3
    add t3, t3, t0
    j next_matrix_char
    
set_matrix_negative:
    li t4, 1
    j next_matrix_char
    
check_matrix_delim:
    li t1, 44
    beq t0, t1, save_matrix_weight
    li t1, 32
    beq t0, t1, save_matrix_weight
    j next_matrix_char
    
save_matrix_weight:
    beqz t3, reset_matrix_vars
    
    bnez t4, make_matrix_negative
    j store_matrix_weight
    
make_matrix_negative:
    sub t3, zero, t3
    
store_matrix_weight:
    # Limita valores para signed byte
    li t5, 127
    blt t3, t5, check_matrix_min
    li t3, 127
    j store_matrix_byte
    
check_matrix_min:
    li t5, -128
    bgt t3, t5, store_matrix_byte
    li t3, -128
    
store_matrix_byte:
    sb t3, 0(s1)
    addi s1, s1, 1
    addi s2, s2, 1
    
    # Limite de segurança
    li t5, 1000
    bge s2, t5, matrix_parse_done
    
reset_matrix_vars:
    li t3, 0
    li t4, 0
    
next_matrix_char:
    addi s0, s0, 1
    j parse_matrix_loop
    
matrix_parse_done:
    mv a0, s2
    
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    addi sp, sp, 16
    ret

# ============================================
# setup_default_weights: Pesos padrão
# ============================================
setup_default_weights:
    addi sp, sp, -4
    sw ra, 0(sp)
    
    # Setup simple identity-like weights
    la t0, weights_l1
    li t1, 0
    li t2, 200
    
default_l1_loop:
    bge t1, t2, setup_l2_default
    li t3, 1
    sb t3, 0(t0)
    addi t0, t0, 1
    addi t1, t1, 1
    j default_l1_loop
    
setup_l2_default:
    la t0, weights_l2
    li t1, 0
    li t2, 1000
    
default_l2_loop:
    bge t1, t2, setup_l3_default
    li t3, 1
    sb t3, 0(t0)
    addi t0, t0, 1
    addi t1, t1, 1
    j default_l2_loop
    
setup_l3_default:
    la t0, weights_l3
    li t1, 0
    li t2, 200
    
default_l3_loop:
    bge t1, t2, setup_l4_default
    li t3, 1
    sb t3, 0(t0)
    addi t0, t0, 1
    addi t1, t1, 1
    j default_l3_loop
    
setup_l4_default:
    la t0, weights_l4
    li t1, 0
    li t2, 30
    
default_l4_loop:
    bge t1, t2, default_weights_done
    li t3, 1
    sb t3, 0(t0)
    addi t0, t0, 1
    addi t1, t1, 1
    j default_l4_loop
    
default_weights_done:
    lw ra, 0(sp)
    addi sp, sp, 4
    ret

# ============================================
# parse_input: Processa entrada com vírgulas
# ============================================
parse_input:
    addi sp, sp, -12
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    
    la s0, input_buffer
    la s1, activation_0
    li t2, 0
    li t3, 0
    
parse_input_loop:
    lb t0, 0(s0)
    beqz t0, save_last_input
    
    li t1, 48
    blt t0, t1, check_input_delimiter
    li t1, 57
    bgt t0, t1, check_input_delimiter
    
    addi t0, t0, -48
    mv t4, t3
    add t3, t4, t4
    add t3, t3, t3
    add t3, t3, t4
    add t3, t3, t3
    add t3, t3, t0
    j next_input_char
    
check_input_delimiter:
    li t1, 44
    beq t0, t1, save_input_number
    li t1, 32
    beq t0, t1, save_input_number
    j next_input_char
    
save_input_number:
    beqz t3, next_input_char
    
    li t4, 255
    blt t3, t4, store_input_byte
    li t3, 255
    
store_input_byte:
    sb t3, 0(s1)
    addi s1, s1, 1
    addi t2, t2, 1
    li t3, 0
    
    li t1, 4
    beq t2, t1, input_done
    
next_input_char:
    addi s0, s0, 1
    j parse_input_loop
    
save_last_input:
    beqz t3, check_input_count
    
    li t4, 255
    blt t3, t4, store_last_byte
    li t3, 255
    
store_last_byte:
    sb t3, 0(s1)
    addi t2, t2, 1
    
check_input_count:
    beqz t2, use_default_input
    j input_done
    
use_default_input:
    la s1, activation_0
    li t0, 4
    sb t0, 0(s1)
    li t0, 50
    sb t0, 1(s1)
    li t0, 20
    sb t0, 2(s1)
    li t0, 10
    sb t0, 3(s1)
    
input_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    addi sp, sp, 12
    ret

# ============================================
# forward_pass_real: Rede neural real com multiplicação de matrizes
# ============================================
forward_pass_real:
    addi sp, sp, -4
    sw ra, 0(sp)
    
    # Layer 1: input (4) -> L1 (50)
    la a0, activation_0     # input
    la a1, weights_l1       # weights
    la a2, activation_1     # output
    li a3, 4               # input_size
    li a4, 50              # output_size
    jal ra, matrix_multiply
    
    # Apply ReLU to L1 output
    la a0, activation_1
    li a1, 50
    jal ra, apply_relu
    
    # Layer 2: L1 (50) -> L2 (20)
    la a0, activation_1
    la a1, weights_l2
    la a2, activation_2
    li a3, 50
    li a4, 20
    jal ra, matrix_multiply
    
    # Apply ReLU to L2 output
    la a0, activation_2
    li a1, 20
    jal ra, apply_relu
    
    # Layer 3: L2 (20) -> L3 (10)
    la a0, activation_2
    la a1, weights_l3
    la a2, activation_3
    li a3, 20
    li a4, 10
    jal ra, matrix_multiply
    
    # Apply ReLU to L3 output
    la a0, activation_3
    li a1, 10
    jal ra, apply_relu
    
    # Layer 4: L3 (10) -> output (3)
    la a0, activation_3
    la a1, weights_l4
    la a2, activation_4
    li a3, 10
    li a4, 3
    jal ra, matrix_multiply
    
    # No ReLU on final output (logits)
    
    lw ra, 0(sp)
    addi sp, sp, 4
    ret

# ============================================
# matrix_multiply: Multiplicação de matriz
# a0 = input vector, a1 = weights matrix, a2 = output vector
# a3 = input_size, a4 = output_size
# ============================================
matrix_multiply:
    addi sp, sp, -20
    sw ra, 0(sp)
    sw s0, 4(sp)
    sw s1, 8(sp)
    sw s2, 12(sp)
    sw s3, 16(sp)
    
    mv s0, a0               # input
    mv s1, a1               # weights
    mv s2, a2               # output
    mv s3, a3               # input_size
    mv t6, a4               # output_size
    
    li t0, 0                # output neuron index
    
multiply_outer_loop:
    bge t0, t6, multiply_done
    
    li t1, 0                # input index
    li t2, 0                # accumulator
    
multiply_inner_loop:
    bge t1, s3, store_result
    
    # Load input[t1]
    add t3, s0, t1
    lb t4, 0(t3)
    
    # Load weight[t0][t1] = weight[t0*input_size + t1]
    mul t3, t0, s3          # t0 * input_size
    add t3, t3, t1          # + t1
    add t3, s1, t3          # + base address
    lb t5, 0(t3)
    
    # Multiply and accumulate (with overflow protection)
    mul t4, t4, t5
    add t2, t2, t4
    
    # Clamp accumulator to prevent overflow
    li t3, 32767
    blt t2, t3, check_min_acc
    li t2, 32767
    j next_multiply_inner
    
check_min_acc:
    li t3, -32768
    bgt t2, t3, next_multiply_inner
    li t3, -32768
    
next_multiply_inner:
    addi t1, t1, 1
    j multiply_inner_loop
    
store_result:
    # Apply scaling to keep values reasonable
    srai t2, t2, 8          # Divide by 256 for scaling
    
    # Clamp to byte range
    li t3, 127
    blt t2, t3, check_min_result
    li t2, 127
    j store_output_byte
    
check_min_result:
    li t3, -128
    bgt t2, t3, store_output_byte
    li t2, -128
    
store_output_byte:
    add t3, s2, t0
    sb t2, 0(t3)
    
    addi t0, t0, 1
    j multiply_outer_loop
    
multiply_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    lw s1, 8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    addi sp, sp, 20
    ret

# ============================================
# apply_relu: Aplica ReLU (max(0, x))
# a0 = vector, a1 = size
# ============================================
apply_relu:
    addi sp, sp, -8
    sw ra, 0(sp)
    sw s0, 4(sp)
    
    mv s0, a0
    li t0, 0
    
relu_loop:
    bge t0, a1, relu_done
    
    add t1, s0, t0
    lb t2, 0(t1)
    
    # Se negativo, coloca 0
    bgez t2, relu_next
    sb zero, 0(t1)
    
relu_next:
    addi t0, t0, 1
    j relu_loop
    
relu_done:
    lw ra, 0(sp)
    lw s0, 4(sp)
    addi sp, sp, 8
    ret

# ============================================
# find_argmax_output: Encontra maior valor na saída
# ============================================
find_argmax_output:
    addi sp, sp, -4
    sw ra, 0(sp)
    
    la t0, activation_4
    li t1, 0                # max_idx
    lb t2, 0(t0)            # max_val
    
    lb t3, 1(t0)
    ble t3, t2, check_output2
    li t1, 1
    mv t2, t3
    
check_output2:
    lb t3, 2(t0)
    ble t3, t2, print_final_result
    li t1, 2
    
print_final_result:
    addi t1, t1, 48
    la t0, result_buffer
    sb t1, 0(t0)
    li t1, 10
    sb t1, 1(t0)
    
    li a0, 1
    la a1, result_buffer
    li a2, 2
    li a7, 64
    ecall
    
    lw ra, 0(sp)
    addi sp, sp, 4
    ret