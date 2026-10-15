import sys, time
sys.argv=[sys.argv[0]]+sys.argv[1:]
import mpmath as mp
exec(open('scan_kklp_charges.py').read().split('if __name__')[0])
th=mp.mpf(sys.argv[1])*mp.pi/180; rs=[mp.mpf(x) for x in sys.argv[2].split(',')]
e1=[2/mp.sqrt(6),-1/mp.sqrt(6),-1/mp.sqrt(6)]; e2=[0,1/mp.sqrt(2),-1/mp.sqrt(2)]
gp=gc=Pg=None; t0=time.time()
for r in rs:
    xi=[r*(mp.cos(th)*a+mp.sin(th)*b) for a,b in zip(e1,e2)]
    p=point(xi,gp,gc,Pg); Q=p['Q']; sQ=sum(Q)
    print('theta=%s r=%.3f Q/sumQ=(%.3f,%.3f,%.3f) Sprot/K=%.4f delta0=%.5f r1=%.4f Phi*/pi=%.4f'%(sys.argv[1],float(r),*(float(q/sQ) for q in Q),float(p['Sprot']),float(p['delta0']),float(p['r1']),float(p['Phi']/mp.pi)),flush=True)
    gp,gc,Pg=(p['mp'][0],p['mp'][1]),(p['mc'][0],p['mc'][1]),p['Phi']
print('%.1fs'%(time.time()-t0))
