Seed-0 entropy-patched BLT, validation only, 95% document bootstrap intervals (2000 resamples). Cache is the mean resident global KV size over positions of a 4,096-byte chunk, computed from entry counts and bit widths, not measured. Eviction rows keep the first patch and the 8 newest. A row is marked * when no other row has equal or fewer bytes and a strictly lower BPB.

## tinystories (full-cache BPB 0.7980)

| Setting | Mean KV cache (KB) | Share of full | BPB [95% CI] | Cost over full |
|---|---|---|---|---|
| evict to 12 patches, recency | 107 | 13% | 0.8314 [0.8180, 0.8447] | +0.0334 |
| evict to 12 patches, random | 107 | 13% | 0.8336 [0.8198, 0.8472] | +0.0356 |
| evict to 12 patches, entropy | 107 | 13% | 0.8376 [0.8240, 0.8511] | +0.0396 |
| evict to 12 patches, entropy_inv * | 107 | 13% | 0.8286 [0.8146, 0.8425] | +0.0306 |
| uniform 2 bits | 118 | 14% | 0.8698 [0.8560, 0.8839] | +0.0718 |
| tiers 4/2 bit, keep low entropy 0.125 high | 127 | 15% | 0.8614 [0.8479, 0.8752] | +0.0634 |
| tiers 4/2 bit, random 0.125 high | 132 | 16% | 0.8619 [0.8482, 0.8758] | +0.0639 |
| tiers 4/2 bit, keep high entropy 0.125 high | 134 | 16% | 0.8664 [0.8527, 0.8805] | +0.0684 |
| tiers 4/2 bit, keep low entropy 0.25 high | 137 | 16% | 0.8492 [0.8356, 0.8626] | +0.0511 |
| evict to 16 patches, recency | 140 | 17% | 0.8214 [0.8079, 0.8347] | +0.0234 |
| evict to 16 patches, random | 140 | 17% | 0.8252 [0.8114, 0.8388] | +0.0272 |
| evict to 16 patches, entropy | 140 | 17% | 0.8329 [0.8193, 0.8463] | +0.0349 |
| evict to 16 patches, entropy_inv * | 140 | 17% | 0.8199 [0.8061, 0.8338] | +0.0219 |
| tiers 4/2 bit, recent 0.125 high * | 144 | 17% | 0.8041 [0.7904, 0.8180] | +0.0061 |
| tiers 4/2 bit, random 0.25 high | 145 | 17% | 0.8524 [0.8386, 0.8664] | +0.0544 |
| tiers 4/2 bit, keep high entropy 0.25 high | 149 | 18% | 0.8609 [0.8470, 0.8749] | +0.0629 |
| tiers 4/2 bit, recent 0.25 high * | 165 | 20% | 0.8021 [0.7883, 0.8159] | +0.0041 |
| tiers 4/2 bit, keep low entropy 0.5 high | 165 | 20% | 0.8258 [0.8123, 0.8394] | +0.0277 |
| uniform 3 bits | 170 | 20% | 0.8028 [0.7894, 0.8162] | +0.0048 |
| tiers 4/2 bit, random 0.5 high | 171 | 20% | 0.8336 [0.8198, 0.8472] | +0.0356 |
| tiers 4/2 bit, keep high entropy 0.5 high | 180 | 21% | 0.8398 [0.8255, 0.8537] | +0.0418 |
| tiers 4/2 bit, recent 0.5 high * | 197 | 24% | 0.8006 [0.7870, 0.8142] | +0.0026 |
| uniform 4 bits * | 223 | 27% | 0.7990 [0.7855, 0.8123] | +0.0010 |
| evict to 32 patches, recency | 266 | 32% | 0.8078 [0.7944, 0.8213] | +0.0098 |
| evict to 32 patches, random | 266 | 32% | 0.8115 [0.7976, 0.8250] | +0.0135 |
| evict to 32 patches, entropy | 266 | 32% | 0.8231 [0.8093, 0.8367] | +0.0250 |
| evict to 32 patches, entropy_inv | 266 | 32% | 0.8089 [0.7951, 0.8226] | +0.0109 |
| uniform 8 bits * | 433 | 52% | 0.7980 [0.7846, 0.8114] | +0.0000 |
| evict to 64 patches, recency | 472 | 56% | 0.8016 [0.7884, 0.8152] | +0.0036 |
| evict to 64 patches, random | 472 | 56% | 0.8033 [0.7897, 0.8169] | +0.0053 |
| evict to 64 patches, entropy | 472 | 56% | 0.8109 [0.7968, 0.8248] | +0.0129 |
| evict to 64 patches, entropy_inv | 472 | 56% | 0.8026 [0.7890, 0.8163] | +0.0046 |
| evict to 128 patches, recency | 712 | 85% | 0.7987 [0.7854, 0.8123] | +0.0007 |
| evict to 128 patches, random | 712 | 85% | 0.7993 [0.7857, 0.8128] | +0.0012 |
| evict to 128 patches, entropy | 712 | 85% | 0.8020 [0.7880, 0.8158] | +0.0040 |
| evict to 128 patches, entropy_inv | 712 | 85% | 0.7993 [0.7856, 0.8129] | +0.0013 |
| full cache fp16 * | 839 | 100% | 0.7980 [0.7846, 0.8114] | +0.0000 |

## wikitext103 (full-cache BPB 1.6801)

| Setting | Mean KV cache (KB) | Share of full | BPB [95% CI] | Cost over full |
|---|---|---|---|---|
| evict to 16 patches, recency * | 145 | 4% | 1.6927 [1.6517, 1.7331] | +0.0126 |
| evict to 16 patches, random | 145 | 4% | 1.6949 [1.6538, 1.7351] | +0.0148 |
| evict to 16 patches, entropy | 145 | 4% | 1.6982 [1.6571, 1.7383] | +0.0182 |
| evict to 16 patches, entropy_inv | 145 | 4% | 1.6964 [1.6545, 1.7373] | +0.0163 |
| evict to 32 patches, recency * | 287 | 8% | 1.6872 [1.6461, 1.7274] | +0.0071 |
| evict to 32 patches, random | 287 | 8% | 1.6900 [1.6491, 1.7301] | +0.0099 |
| evict to 32 patches, entropy | 287 | 8% | 1.6957 [1.6550, 1.7357] | +0.0157 |
| evict to 32 patches, entropy_inv | 287 | 8% | 1.6932 [1.6513, 1.7342] | +0.0131 |
| uniform 2 bits | 531 | 14% | 1.7045 [1.6641, 1.7441] | +0.0244 |
| evict to 64 patches, recency * | 558 | 15% | 1.6836 [1.6427, 1.7236] | +0.0035 |
| evict to 64 patches, random | 558 | 15% | 1.6860 [1.6452, 1.7260] | +0.0060 |
| evict to 64 patches, entropy | 558 | 15% | 1.6922 [1.6518, 1.7321] | +0.0122 |
| evict to 64 patches, entropy_inv | 558 | 15% | 1.6901 [1.6483, 1.7306] | +0.0100 |
| tiers 4/2 bit, random 0.125 high | 592 | 16% | 1.7019 [1.6614, 1.7414] | +0.0218 |
| tiers 4/2 bit, keep low entropy 0.125 high | 592 | 16% | 1.7007 [1.6603, 1.7404] | +0.0207 |
| tiers 4/2 bit, keep high entropy 0.125 high | 593 | 16% | 1.7031 [1.6628, 1.7427] | +0.0230 |
| tiers 4/2 bit, recent 0.125 high | 643 | 17% | 1.6836 [1.6430, 1.7234] | +0.0036 |
| tiers 4/2 bit, random 0.25 high | 650 | 17% | 1.6992 [1.6587, 1.7388] | +0.0191 |
| tiers 4/2 bit, keep low entropy 0.25 high | 651 | 17% | 1.6971 [1.6567, 1.7365] | +0.0170 |
| tiers 4/2 bit, keep high entropy 0.25 high | 651 | 17% | 1.7017 [1.6615, 1.7411] | +0.0216 |
| tiers 4/2 bit, recent 0.25 high * | 739 | 20% | 1.6829 [1.6422, 1.7226] | +0.0028 |
| uniform 3 bits * | 767 | 20% | 1.6820 [1.6413, 1.7217] | +0.0019 |
| tiers 4/2 bit, random 0.5 high | 768 | 20% | 1.6937 [1.6530, 1.7332] | +0.0136 |
| tiers 4/2 bit, keep high entropy 0.5 high | 769 | 20% | 1.6972 [1.6568, 1.7370] | +0.0171 |
| tiers 4/2 bit, keep low entropy 0.5 high | 770 | 20% | 1.6902 [1.6497, 1.7300] | +0.0102 |
| tiers 4/2 bit, recent 0.5 high * | 886 | 23% | 1.6815 [1.6408, 1.7213] | +0.0014 |
| uniform 4 bits * | 1,003 | 27% | 1.6805 [1.6398, 1.7202] | +0.0004 |
| evict to 128 patches, recency | 1,059 | 28% | 1.6814 [1.6407, 1.7213] | +0.0013 |
| evict to 128 patches, random | 1,059 | 28% | 1.6833 [1.6425, 1.7232] | +0.0032 |
| evict to 128 patches, entropy | 1,059 | 28% | 1.6885 [1.6480, 1.7281] | +0.0084 |
| evict to 128 patches, entropy_inv | 1,059 | 28% | 1.6864 [1.6448, 1.7267] | +0.0063 |
| evict to 256 patches, recency | 1,915 | 51% | 1.6805 [1.6398, 1.7203] | +0.0004 |
| evict to 256 patches, random | 1,915 | 51% | 1.6814 [1.6407, 1.7213] | +0.0013 |
| evict to 256 patches, entropy | 1,915 | 51% | 1.6841 [1.6437, 1.7237] | +0.0041 |
| evict to 256 patches, entropy_inv | 1,915 | 51% | 1.6827 [1.6415, 1.7229] | +0.0027 |
| uniform 8 bits * | 1,948 | 52% | 1.6801 [1.6394, 1.7199] | -0.0000 |
| evict to 512 patches, recency | 3,105 | 82% | 1.6801 [1.6395, 1.7199] | +0.0000 |
| evict to 512 patches, random | 3,105 | 82% | 1.6804 [1.6397, 1.7202] | +0.0003 |
| evict to 512 patches, entropy | 3,105 | 82% | 1.6809 [1.6403, 1.7207] | +0.0008 |
| evict to 512 patches, entropy_inv | 3,105 | 82% | 1.6806 [1.6396, 1.7204] | +0.0005 |
| full cache fp16 | 3,777 | 100% | 1.6801 [1.6394, 1.7199] | +0.0000 |

## tinystoriesmix (full-cache BPB 0.9019)

| Setting | Mean KV cache (KB) | Share of full | BPB [95% CI] | Cost over full |
|---|---|---|---|---|
| evict to 12 patches, recency | 107 | 13% | 0.9213 [0.9073, 0.9356] | +0.0194 |
| evict to 12 patches, random | 107 | 13% | 0.9201 [0.9059, 0.9345] | +0.0181 |
| evict to 12 patches, entropy | 107 | 13% | 0.9246 [0.9108, 0.9391] | +0.0227 |
| evict to 12 patches, entropy_inv * | 107 | 13% | 0.9144 [0.9003, 0.9289] | +0.0125 |
| uniform 2 bits | 112 | 14% | 0.9315 [0.9178, 0.9458] | +0.0296 |
| tiers 4/2 bit, keep low entropy 0.125 high | 121 | 15% | 0.9270 [0.9132, 0.9412] | +0.0251 |
| tiers 4/2 bit, random 0.125 high | 126 | 16% | 0.9277 [0.9138, 0.9420] | +0.0258 |
| tiers 4/2 bit, keep high entropy 0.125 high | 127 | 16% | 0.9297 [0.9159, 0.9441] | +0.0278 |
| tiers 4/2 bit, keep low entropy 0.25 high | 131 | 16% | 0.9235 [0.9095, 0.9376] | +0.0215 |
| tiers 4/2 bit, recent 0.125 high * | 137 | 17% | 0.9050 [0.8911, 0.9192] | +0.0031 |
| tiers 4/2 bit, random 0.25 high | 138 | 17% | 0.9251 [0.9112, 0.9393] | +0.0231 |
| tiers 4/2 bit, keep high entropy 0.25 high | 140 | 18% | 0.9276 [0.9138, 0.9419] | +0.0256 |
| evict to 16 patches, recency | 140 | 18% | 0.9163 [0.9023, 0.9305] | +0.0143 |
| evict to 16 patches, random | 140 | 18% | 0.9150 [0.9008, 0.9295] | +0.0130 |
| evict to 16 patches, entropy | 140 | 18% | 0.9224 [0.9084, 0.9369] | +0.0205 |
| evict to 16 patches, entropy_inv | 140 | 18% | 0.9102 [0.8961, 0.9247] | +0.0082 |
| tiers 4/2 bit, recent 0.25 high * | 157 | 20% | 0.9042 [0.8904, 0.9184] | +0.0023 |
| tiers 4/2 bit, keep low entropy 0.5 high | 158 | 20% | 0.9143 [0.9004, 0.9286] | +0.0123 |
| uniform 3 bits | 162 | 20% | 0.9047 [0.8909, 0.9190] | +0.0028 |
| tiers 4/2 bit, random 0.5 high | 163 | 20% | 0.9180 [0.9042, 0.9322] | +0.0160 |
| tiers 4/2 bit, keep high entropy 0.5 high | 170 | 21% | 0.9203 [0.9063, 0.9348] | +0.0183 |
| tiers 4/2 bit, recent 0.5 high * | 187 | 24% | 0.9034 [0.8895, 0.9175] | +0.0014 |
| uniform 4 bits * | 212 | 27% | 0.9024 [0.8886, 0.9165] | +0.0005 |
| evict to 32 patches, recency | 264 | 33% | 0.9091 [0.8953, 0.9234] | +0.0072 |
| evict to 32 patches, random | 264 | 33% | 0.9074 [0.8934, 0.9218] | +0.0054 |
| evict to 32 patches, entropy | 264 | 33% | 0.9167 [0.9025, 0.9312] | +0.0147 |
| evict to 32 patches, entropy_inv | 264 | 33% | 0.9055 [0.8915, 0.9198] | +0.0035 |
| uniform 8 bits * | 411 | 52% | 0.9019 [0.8881, 0.9160] | +0.0000 |
| evict to 64 patches, recency | 466 | 58% | 0.9044 [0.8906, 0.9184] | +0.0024 |
| evict to 64 patches, random | 466 | 58% | 0.9036 [0.8898, 0.9177] | +0.0017 |
| evict to 64 patches, entropy | 466 | 58% | 0.9090 [0.8947, 0.9234] | +0.0071 |
| evict to 64 patches, entropy_inv | 466 | 58% | 0.9035 [0.8896, 0.9177] | +0.0016 |
| evict to 128 patches, recency | 689 | 86% | 0.9022 [0.8883, 0.9162] | +0.0002 |
| evict to 128 patches, random | 689 | 86% | 0.9022 [0.8884, 0.9163] | +0.0003 |
| evict to 128 patches, entropy | 689 | 86% | 0.9040 [0.8899, 0.9184] | +0.0021 |
| evict to 128 patches, entropy_inv | 689 | 86% | 0.9024 [0.8885, 0.9166] | +0.0004 |
| full cache fp16 * | 797 | 100% | 0.9019 [0.8881, 0.9160] | +0.0000 |

## wikitext103mix (full-cache BPB 1.7190)

| Setting | Mean KV cache (KB) | Share of full | BPB [95% CI] | Cost over full |
|---|---|---|---|---|
| evict to 16 patches, recency * | 145 | 4% | 1.7272 [1.6881, 1.7657] | +0.0082 |
| evict to 16 patches, random | 145 | 4% | 1.7296 [1.6905, 1.7682] | +0.0106 |
| evict to 16 patches, entropy | 145 | 4% | 1.7311 [1.6920, 1.7698] | +0.0121 |
| evict to 16 patches, entropy_inv | 145 | 4% | 1.7306 [1.6913, 1.7695] | +0.0116 |
| evict to 32 patches, recency * | 287 | 8% | 1.7233 [1.6845, 1.7617] | +0.0043 |
| evict to 32 patches, random | 287 | 8% | 1.7266 [1.6878, 1.7651] | +0.0076 |
| evict to 32 patches, entropy | 287 | 8% | 1.7288 [1.6898, 1.7675] | +0.0098 |
| evict to 32 patches, entropy_inv | 287 | 8% | 1.7285 [1.6892, 1.7673] | +0.0095 |
| uniform 2 bits | 531 | 14% | 1.7508 [1.7128, 1.7888] | +0.0318 |
| evict to 64 patches, recency * | 558 | 15% | 1.7211 [1.6825, 1.7593] | +0.0021 |
| evict to 64 patches, random | 558 | 15% | 1.7243 [1.6855, 1.7628] | +0.0053 |
| evict to 64 patches, entropy | 558 | 15% | 1.7264 [1.6876, 1.7647] | +0.0074 |
| evict to 64 patches, entropy_inv | 558 | 15% | 1.7266 [1.6873, 1.7655] | +0.0076 |
| tiers 4/2 bit, random 0.125 high | 592 | 16% | 1.7475 [1.7094, 1.7855] | +0.0285 |
| tiers 4/2 bit, keep low entropy 0.125 high | 592 | 16% | 1.7461 [1.7081, 1.7842] | +0.0271 |
| tiers 4/2 bit, keep high entropy 0.125 high | 594 | 16% | 1.7494 [1.7114, 1.7874] | +0.0304 |
| tiers 4/2 bit, recent 0.125 high | 643 | 17% | 1.7252 [1.6866, 1.7634] | +0.0062 |
| tiers 4/2 bit, random 0.25 high | 650 | 17% | 1.7439 [1.7058, 1.7820] | +0.0249 |
| tiers 4/2 bit, keep low entropy 0.25 high | 651 | 17% | 1.7415 [1.7035, 1.7795] | +0.0225 |
| tiers 4/2 bit, keep high entropy 0.25 high | 652 | 17% | 1.7475 [1.7096, 1.7854] | +0.0285 |
| tiers 4/2 bit, recent 0.25 high | 739 | 20% | 1.7242 [1.6857, 1.7624] | +0.0052 |
| uniform 3 bits * | 767 | 20% | 1.7207 [1.6822, 1.7588] | +0.0017 |
| tiers 4/2 bit, random 0.5 high | 768 | 20% | 1.7365 [1.6983, 1.7746] | +0.0175 |
| tiers 4/2 bit, keep high entropy 0.5 high | 769 | 20% | 1.7409 [1.7024, 1.7791] | +0.0219 |
| tiers 4/2 bit, keep low entropy 0.5 high | 770 | 20% | 1.7326 [1.6943, 1.7706] | +0.0136 |
| tiers 4/2 bit, recent 0.5 high | 886 | 23% | 1.7216 [1.6830, 1.7597] | +0.0025 |
| uniform 4 bits * | 1,003 | 27% | 1.7193 [1.6808, 1.7574] | +0.0003 |
| evict to 128 patches, recency | 1,059 | 28% | 1.7197 [1.6812, 1.7578] | +0.0007 |
| evict to 128 patches, random | 1,059 | 28% | 1.7223 [1.6837, 1.7606] | +0.0033 |
| evict to 128 patches, entropy | 1,059 | 28% | 1.7240 [1.6856, 1.7623] | +0.0050 |
| evict to 128 patches, entropy_inv | 1,059 | 28% | 1.7245 [1.6853, 1.7632] | +0.0055 |
| evict to 256 patches, recency * | 1,915 | 51% | 1.7192 [1.6807, 1.7572] | +0.0002 |
| evict to 256 patches, random | 1,915 | 51% | 1.7206 [1.6820, 1.7587] | +0.0016 |
| evict to 256 patches, entropy | 1,915 | 51% | 1.7216 [1.6831, 1.7597] | +0.0026 |
| evict to 256 patches, entropy_inv | 1,915 | 51% | 1.7218 [1.6830, 1.7603] | +0.0028 |
| uniform 8 bits * | 1,948 | 52% | 1.7190 [1.6805, 1.7571] | +0.0000 |
| evict to 512 patches, recency | 3,107 | 82% | 1.7190 [1.6806, 1.7571] | +0.0000 |
| evict to 512 patches, random | 3,107 | 82% | 1.7194 [1.6809, 1.7574] | +0.0004 |
| evict to 512 patches, entropy | 3,107 | 82% | 1.7196 [1.6810, 1.7576] | +0.0006 |
| evict to 512 patches, entropy_inv | 3,107 | 82% | 1.7195 [1.6810, 1.7576] | +0.0005 |
| full cache fp16 * | 3,778 | 100% | 1.7190 [1.6805, 1.7571] | +0.0000 |
