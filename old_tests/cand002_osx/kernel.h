// This file automatically generated by template-converter
// DO NOT EDIT!

const char *kern =
"#pragma OPENCL EXTENSION cl_amd_printf : enable\n"
"\n"
"//HARDCODE ALL OF THESE!\n"
"#define N_ANT  256u  //2560u\n"
"#define N_BLK   36u  //3240u // N(N+1)/2 where N=(N_ANT/32)\n"
"#define N_ANT_4 64u  //640u  // N/4\n"
"\n"
"\n"
"\n"
"__kernel void corr(__global uint *packed,\n"
"	           __global  int *corr_buf,\n"
"		   __global uint *id_x_map,\n"
"		   __global uint *id_y_map,\n"
"		   __local  uint *unpacked)\n"
"{\n"
"  uint lx = get_local_id(0);\n"
"  uint ly = get_local_id(1);\n"
"\n"
"  uint z = get_global_id(2) / N_BLK;\n"
"  uint blkid = get_global_id(2) - z*N_BLK;\n"
"  uint x = id_x_map[get_global_id(2)-z*N_BLK];\n"
"  uint y = id_y_map[get_global_id(2)-z*N_BLK];\n"
"\n"
"  uint q = (lx % 2) * 16;\n"
"  uint addr_x = ((get_local_size(0)*x+lx)*2  + z*N_ANT*256) / 4;\n"
"  uint addr_y = ((y*32 + get_global_id(1)*4) + z*N_ANT*256 + lx*N_ANT)/4;\n"
"\n"
"  uint corr_a[4]={0,0,0,0};\n"
"  uint corr_b[4]={0,0,0,0};\n"
"  uint corr_c[4]={0,0,0,0};\n"
"  uint corr_d[4]={0,0,0,0};\n"
"\n"
"  uint offs[4]={q,q+4,q+8,q+12};\n"
"\n"
"  for (uint i=0; i<256; i+=get_local_size(0))\n"
"  {\n"
"    uint la=(lx*16 + ly*4);\n"
"    uint pa=packed[mad24(i,N_ANT_4,addr_y)];\n"
"\n"
"    unpacked[la]   = (((pa >>  0) & 0xf) << 16) + ((pa >>  4) & 0xf);\n"
"    unpacked[la+1] = (((pa >>  8) & 0xf) << 16) + ((pa >> 12) & 0xf);\n"
"    unpacked[la+2] = (((pa >> 16) & 0xf) << 16) + ((pa >> 20) & 0xf);\n"
"    unpacked[la+3] = (((pa >> 24) & 0xf) << 16) + ((pa >> 28) & 0xf);\n"
"\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"
"    for (uint j=0; j<get_local_size(0); j++)\n"
"    {\n"
"      uint temp_ua = unpacked[j*16+ly*4];\n"
"      uint temp_ub = unpacked[j*16+ly*4+1];\n"
"      uint temp_uc = unpacked[j*16+ly*4+2];\n"
"      uint temp_ud = unpacked[j*16+ly*4+3];\n"
"\n"
"      uint temp_par = (packed[addr_x+(i+j)*N_ANT_4] >> offs[0]) & 0xf;\n"
"      uint temp_pai = (packed[addr_x+(i+j)*N_ANT_4] >> offs[1]) & 0xf;\n"
"      uint temp_pbr = (packed[addr_x+(i+j)*N_ANT_4] >> offs[2]) & 0xf;\n"
"      uint temp_pbi = (packed[addr_x+(i+j)*N_ANT_4] >> offs[3]) & 0xf;\n"
"\n"
"      corr_a[0]=mad24(temp_par, temp_ua, corr_a[0]);\n"
"      corr_a[1]=mad24(temp_pai, temp_ua, corr_a[1]);\n"
"      corr_a[2]=mad24(temp_pbr, temp_ua, corr_a[2]);\n"
"      corr_a[3]=mad24(temp_pbi, temp_ua, corr_a[3]);\n"
"\n"
"      corr_b[0]=mad24(temp_par, temp_ub, corr_b[0]);\n"
"      corr_b[1]=mad24(temp_pai, temp_ub, corr_b[1]);\n"
"      corr_b[2]=mad24(temp_pbr, temp_ub, corr_b[2]);\n"
"      corr_b[3]=mad24(temp_pbi, temp_ub, corr_b[3]);\n"
"\n"
"      corr_c[0]=mad24(temp_par, temp_uc, corr_c[0]);\n"
"      corr_c[1]=mad24(temp_pai, temp_uc, corr_c[1]);\n"
"      corr_c[2]=mad24(temp_pbr, temp_uc, corr_c[2]);\n"
"      corr_c[3]=mad24(temp_pbi, temp_uc, corr_c[3]);\n"
"\n"
"      corr_d[0]=mad24(temp_par, temp_ud, corr_d[0]);\n"
"      corr_d[1]=mad24(temp_pai, temp_ud, corr_d[1]);\n"
"      corr_d[2]=mad24(temp_pbr, temp_ud, corr_d[2]);\n"
"      corr_d[3]=mad24(temp_pbi, temp_ud, corr_d[3]);\n"
"    }\n"
"  }\n"
"\n"
"  uint addr_o = ((blkid * 1024) + (get_global_id(1) * 4 * 32) + (lx * 2)) * 2;\n"
"\n"
"  atomic_add(&corr_buf[addr_o],   (corr_a[0] >> 16) + (corr_a[1] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+1], (corr_a[1] >> 16) - (corr_a[0] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+2], (corr_a[2] >> 16) + (corr_a[3] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+3], (corr_a[3] >> 16) - (corr_a[2] & 0xffff) );\n"
"\n"
"  atomic_add(&corr_buf[addr_o+64], (corr_b[0] >> 16) + (corr_b[1] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+65], (corr_b[1] >> 16) - (corr_b[0] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+66], (corr_b[2] >> 16) + (corr_b[3] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+67], (corr_b[3] >> 16) - (corr_b[2] & 0xffff) );\n"
"\n"
"  atomic_add(&corr_buf[addr_o+128], (corr_c[0] >> 16) + (corr_c[1] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+129], (corr_c[1] >> 16) - (corr_c[0] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+130], (corr_c[2] >> 16) + (corr_c[3] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+131], (corr_c[3] >> 16) - (corr_c[2] & 0xffff) );\n"
"\n"
"  atomic_add(&corr_buf[addr_o+192], (corr_d[0] >> 16) + (corr_d[1] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+193], (corr_d[1] >> 16) - (corr_d[0] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+194], (corr_d[2] >> 16) + (corr_d[3] & 0xffff) );\n"
"  atomic_add(&corr_buf[addr_o+195], (corr_d[3] >> 16) - (corr_d[2] & 0xffff) );\n"
"}\n"
;
