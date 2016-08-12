1. Semi-adaptive method: TS1-s1

TS1_s1:  
    original version 
    similar with FPCA
   
   
TS1_s1_v2:
    use lineartimesvd to approximate singular values
    ad: fast
    con: sometimes not so accurate         
    this version is the fastest among all TS1
    
    
TS1_s1_v3:
    use randsvd to approximate singular values
    ad: faster than svds or svd
        more accurate than lineartimesvd
    con: slower than version 2
    
Note: recommend to use TS1_s1_v2 first
      if not convergenced, try TS1_s1_v3




2. Adaptive method: TS1-s2

TS1_s2
    similar with TS1_s1_v2
    use lineartimesvd to approximate singular values


