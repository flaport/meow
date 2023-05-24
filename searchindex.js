Search.setIndex({docnames:["examples","examples/00_introduction","examples/01_gds_taper","examples/02_taper_length_sweep","index","meow","meow.eme","meow.fde"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["examples.md","examples/00_introduction.ipynb","examples/01_gds_taper.ipynb","examples/02_taper_length_sweep.ipynb","index.md","meow.rst","meow.eme.rst","meow.fde.rst"],objects:{"":[[5,0,0,"-","meow"]],"meow.base_model":[[5,1,1,"","BaseModel"]],"meow.base_model.BaseModel":[[5,2,1,"","__init__"]],"meow.cache":[[5,3,1,"","cache_array"],[5,3,1,"","cache_model"],[5,3,1,"","disable_cache"],[5,3,1,"","empty_cache"],[5,3,1,"","enable_cache"]],"meow.cell":[[5,1,1,"","Cell"],[5,3,1,"","create_cells"]],"meow.cell.Cell":[[5,2,1,"","__init__"],[5,4,1,"","extra"],[5,5,1,"","length"],[5,5,1,"","materials"],[5,4,1,"","mesh"],[5,5,1,"","mx"],[5,5,1,"","my"],[5,5,1,"","mz"],[5,4,1,"","structures"],[5,5,1,"","z"],[5,4,1,"","z_max"],[5,4,1,"","z_min"]],"meow.cross_section":[[5,1,1,"","CrossSection"]],"meow.cross_section.CrossSection":[[5,2,1,"","__init__"],[5,4,1,"","cell"],[5,4,1,"","env"],[5,4,1,"","extra"],[5,5,1,"","mesh"],[5,5,1,"","nx"],[5,5,1,"","ny"],[5,5,1,"","nz"],[5,5,1,"","structures"]],"meow.eme":[[6,0,0,"-","common"],[6,0,0,"-","sax"]],"meow.eme.common":[[6,3,1,"","compute_interface_s_matrices"],[6,3,1,"","compute_interface_s_matrix"],[6,3,1,"","compute_propagation_s_matrices"],[6,3,1,"","compute_propagation_s_matrix"],[6,3,1,"","select_ports"]],"meow.eme.sax":[[6,3,1,"","compute_s_matrix_sax"]],"meow.environment":[[5,1,1,"","Environment"]],"meow.environment.Environment":[[5,4,1,"","T"],[5,4,1,"","wl"]],"meow.fde":[[7,0,0,"-","lumerical"],[7,0,0,"-","tidy3d"]],"meow.fde.lumerical":[[7,3,1,"","compute_modes_lumerical"],[7,3,1,"","get_sim"],[7,3,1,"","set_sim"]],"meow.fde.tidy3d":[[7,3,1,"","compute_modes_tidy3d"]],"meow.gds_structures":[[5,1,1,"","GdsExtrusionRule"],[5,3,1,"","extrude_gds"]],"meow.gds_structures.GdsExtrusionRule":[[5,4,1,"","buffer"],[5,4,1,"","h_max"],[5,4,1,"","h_min"],[5,4,1,"","material"],[5,4,1,"","mesh_order"]],"meow.geometries":[[5,1,1,"","Box"],[5,1,1,"","Geometry"],[5,1,1,"","Prism"]],"meow.geometries.Box":[[5,4,1,"","x_max"],[5,4,1,"","x_min"],[5,4,1,"","y_max"],[5,4,1,"","y_min"],[5,4,1,"","z_max"],[5,4,1,"","z_min"]],"meow.geometries.Geometry":[[5,4,1,"","type"]],"meow.geometries.Prism":[[5,4,1,"","axis"],[5,4,1,"","h_max"],[5,4,1,"","h_min"],[5,4,1,"","poly"]],"meow.integrate":[[5,3,1,"","integrate_2d"],[5,3,1,"","integrate_interpolate_2d"]],"meow.materials":[[5,1,1,"","Material"]],"meow.materials.Material":[[5,2,1,"","__init__"],[5,2,1,"","from_df"],[5,2,1,"","from_path"],[5,4,1,"","meta"],[5,4,1,"","n"],[5,4,1,"","name"],[5,4,1,"","params"]],"meow.mesh":[[5,1,1,"","Mesh"],[5,1,1,"","Mesh2d"]],"meow.mesh.Mesh2d":[[5,5,1,"","X"],[5,5,1,"","Xx"],[5,5,1,"","Xy"],[5,5,1,"","Xz"],[5,5,1,"","Y"],[5,5,1,"","Yx"],[5,5,1,"","Yy"],[5,5,1,"","Yz"],[5,4,1,"","angle_phi"],[5,4,1,"","angle_theta"],[5,4,1,"","bend_axis"],[5,4,1,"","bend_radius"],[5,5,1,"","dx"],[5,5,1,"","dy"],[5,4,1,"","num_pml"],[5,4,1,"","x"],[5,5,1,"","x_"],[5,4,1,"","y"],[5,5,1,"","y_"]],"meow.mode":[[5,1,1,"","Mode"],[5,3,1,"","electric_energy"],[5,3,1,"","electric_energy_density"],[5,3,1,"","energy"],[5,3,1,"","energy_density"],[5,3,1,"","inner_product"],[5,3,1,"","invert_mode"],[5,3,1,"","magnetic_energy"],[5,3,1,"","magnetic_energy_density"],[5,3,1,"","normalize_energy"],[5,3,1,"","normalize_product"],[5,3,1,"","te_fraction"],[5,3,1,"","zero_phase"]],"meow.mode.Mode":[[5,5,1,"","A"],[5,4,1,"","Ex"],[5,4,1,"","Ey"],[5,4,1,"","Ez"],[5,4,1,"","Hx"],[5,4,1,"","Hy"],[5,4,1,"","Hz"],[5,5,1,"","Px"],[5,5,1,"","Py"],[5,5,1,"","Pz"],[5,2,1,"","__init__"],[5,5,1,"","cell"],[5,4,1,"","cs"],[5,5,1,"","env"],[5,2,1,"","load"],[5,5,1,"","mesh"],[5,4,1,"","neff"],[5,2,1,"","save"],[5,5,1,"","te_fraction"]],"meow.structures":[[5,1,1,"","Structure"],[5,3,1,"","sort_structures"],[5,3,1,"","visualize_structures"]],"meow.structures.Structure":[[5,4,1,"","geometry"],[5,4,1,"","material"],[5,4,1,"","mesh_order"]],"meow.visualize":[[5,3,1,"","vis"],[5,3,1,"","visualize"]],meow:[[5,0,0,"-","base_model"],[5,0,0,"-","cache"],[5,0,0,"-","cell"],[5,0,0,"-","cross_section"],[6,0,0,"-","eme"],[5,0,0,"-","environment"],[7,0,0,"-","fde"],[5,0,0,"-","gds_structures"],[5,0,0,"-","geometries"],[5,0,0,"-","integrate"],[5,0,0,"-","materials"],[5,0,0,"-","mesh"],[5,0,0,"-","mode"],[5,0,0,"-","structures"],[5,0,0,"-","visualize"]]},objnames:{"0":["py","module","Python module"],"1":["py","pydantic_model","Python model"],"2":["py","method","Python method"],"3":["py","function","Python function"],"4":["py","pydantic_field","Python field"],"5":["py","property","Python property"]},objtypes:{"0":"py:module","1":"py:pydantic_model","2":"py:method","3":"py:function","4":"py:pydantic_field","5":"py:property"},terms:{"0":[1,2,3,5],"020":2,"05":[1,2,3],"06":7,"1":[2,5],"10":[1,2,3,7],"100":[1,2,3],"101":[1,3],"13":2,"17":2,"1e":7,"2":[2,5],"20":2,"21":2,"22":[1,3],"220":2,"24":3,"25":[1,2,3,5],"2d":[1,5],"3":2,"353":[1,2,3],"360":[1,2,3],"39":[2,3],"4":2,"40":[1,2],"45":[1,2,3],"450":2,"5":[2,5],"500px":[1,2,3],"51":2,"55":[1,2,3],"6":[1,2],"7":[1,2],"75":2,"84":2,"95":2,"case":5,"class":5,"default":[1,2,3,5,6,7],"enum":5,"final":[4,6],"float":[2,5,7],"function":1,"import":[1,2],"int":[5,6],"new":5,"return":[2,3,5,6,7],"true":[1,2,3,6],"try":1,A:[1,4,5],For:5,No:[1,2,3],That:1,The:[1,7],_:[1,2,3,5],__init__:[2,5],_visual:5,abov:[1,4],accept:5,accord:5,account:1,add:5,additionalproperti:5,all:[1,4,5,6],allof:5,along:5,also:1,altern:4,although:4,alwai:5,an:[1,4,5,6],angl:5,angle_phi:5,angle_theta:5,ani:[1,4,5,6],anoth:5,anyof:5,append:[1,3],appli:5,approxim:1,ar:[1,5],area:5,arg:2,argument:5,arr:5,arrai:[1,3,5],associ:5,attribut:[1,2,3,5],auto_widen:2,avail:[1,2,3],aveguid:4,ax:5,axi:[1,2,3,5],azimuth:5,back:[1,2,3],backend:[1,2,3,4,6,7],base:[4,5],base_model:5,baseclass:5,basemodel:5,bbox:2,belong:6,bend:5,bend_axi:[1,3,5],bend_radiu:[1,3,5],best:5,better:5,between:[5,6,7],bienstman:4,bool:6,border:[1,2,3],box:[1,5],buffer:[2,5],c988da3e:2,cache_arrai:5,cache_model:5,calcul:[4,5,6,7],can:[1,4,5],cannot:5,cascad:4,cell:[4,5,6],cell_length:3,center:5,certain:[5,6],chang:2,check:5,chip:[1,2,5],circuit:[1,4],clad:2,clad_width:2,classmethod:5,client:2,collect:[1,5,6],combin:[1,5],common:5,commonli:5,compil:[1,2,3],complex:5,compon:[1,2,4],comput:[1,4,7],compute_interface_s_matric:6,compute_interface_s_matrix:6,compute_mod:[1,2,3],compute_modes_lumer:7,compute_modes_tidy3d:7,compute_propagation_s_matric:6,compute_propagation_s_matrix:6,compute_s_matrix:[1,2,3],compute_s_matrix_sax:6,concurr:1,configur:5,conform:[1,3],confusingli:2,constant:1,constraint:5,contain:[1,5],convent:2,convers:7,coord:5,coordin:[2,5],core:[1,2,3],core_width:2,corner:5,cpu:[1,2,3],creat:[1,2,5],create_cel:[1,2,3,5],create_cross_sect:3,create_structur:3,creation:4,cross:[4,5],cross_sect:[2,5,7],crosssect:[1,2,3,5,6,7],cs:[1,2,3,5,7],css:[1,2,3],cuboid:5,cuda:[1,2,3],current:5,curvatur:5,custom:5,dall:4,data:5,datatyp:5,dblquad:5,def:[2,3],defin:5,definit:5,denot:2,densiti:5,depend:[4,5],deriv:5,describ:5,descript:5,dev:4,df:5,dict:[5,6],dictionari:5,diff:2,direct:[1,2,5],disable_cach:5,discret:5,discrit:5,divid:[1,4],doe:1,don:5,downselect:6,draw:4,driver:[1,2,3],dtype:[5,6],dure:4,dx:5,dy:5,e:[4,5],each:6,easili:[1,5],effect:[5,7],effici:4,eigenmod:[4,5],either:5,electr:5,electric_energi:5,electric_energy_dens:5,electromagnet:4,em:[4,5],emepi:4,empty_cach:5,enabl:[1,2,3,4],enable_cach:5,enable_plugin_devic:[1,2,3],end:5,energi:5,energy_dens:5,enforce_lossy_unitar:6,enforce_reciproc:6,env:[1,2,3,5],environ:[1,2,3,5],equival:4,everywher:5,ex:[1,5],exampl:5,example_extrus:2,example_gds_cross_sect:2,example_tap:2,example_taper_length20:2,excel:4,expans:4,expect:[1,2],explicitli:4,extent:5,extra:5,extrud:[1,4,5],extrude_gd:[2,5],extrus:[2,5],extrusion_rul:2,extruson:5,ey:[4,5],ez:5,factor:7,fall:[1,2,3],fals:[1,2,3,6],fde:[4,5],fdespec:7,feature1:4,feature2:4,field:[1,2,5],file:1,filenam:5,find:4,find_mod:3,find_s_matrix:3,first:[1,5],float64:[5,6],follow:4,form:5,found:[1,2,3],fraction:5,free:4,from:[1,2,4,5,6],from_df:5,from_path:5,full:5,fulli:5,futur:2,gd:[0,1,4,5],gds_structur:5,gdsextrusionrul:[1,2,5],gdsfactori:[2,4],gdspy:5,gdstk:5,geometri:[1,3,5],get:[5,6],get_plugin_device_cli:[1,2,3],get_sim:7,gf:2,given:[1,2,3,5,6,7],global:5,gpu:[1,2,3],gpuallocatorconfig:[1,2,3],grid:5,grow:5,guess:7,h:5,h_max:[2,3,5],h_min:[2,3,5],h_sim:2,ha:[1,2,3,5],half:5,handl:5,have:[1,5],header:4,height:[1,2,3,5],help:5,henc:2,here:[1,3],hint:5,hood:1,how:5,howev:1,hx:[1,2,5],hy:5,hz:5,i:[2,5],idx:1,ifram:[1,2,3],igenmod:4,ignor:2,implement:[6,7],impli:4,in0:2,index:[1,4,5,6,7],inf:5,infin:5,influenc:1,info:[1,2,3],inform:[1,5],initi:[1,2,3,7],inject:5,inner:5,inner_product:5,input:[2,5],input_c:2,instal:2,integ:5,integr:5,integrate_2d:5,integrate_interpolate_2d:5,interest:5,interfac:6,interpol:5,introduct:[1,3],invalid_argu:[1,2,3],invert:5,invert_mod:5,invit:1,ipi:4,ipython:4,item:[2,5],its:[4,5],itself:5,jax:4,jaxlib:[1,2,3],json:5,jupyt:4,keep:[1,6],keyword:5,kitten:4,klu:4,klujax:4,kwarg:[5,6],laser:4,later:5,layer:[2,5],left:[1,2],length:[1,2,3,5,6],li:5,librari:1,linear:2,linspac:[1,2,3],list:[1,3,5,6,7],liter:5,load:5,locat:5,logo:4,lower:2,ls:[1,2,3,5],lumapi:7,lumer:7,m:[4,7],magnet:5,magnetic_energi:5,magnetic_energy_dens:5,main:5,make:[1,2],map:[4,6],materi:[1,2,3,5],matplotlib:[2,4],matric:[4,6],matrix:[4,6],max:5,maximum:5,maxitem:5,me:4,meow:[1,2,3],mesh2d:[1,2,3,5],mesh:[1,2,3,5],mesh_ord:[2,5],meta:5,metadata:5,method:5,min:[4,5],mind:1,minimum:5,minitem:5,mode1:5,mode2:5,mode:[4,5,6,7],model:[4,5],modes1:6,modes2:6,modes_in_c:[1,3],modul:[1,2,3,4],more:[1,2,3],most:5,much:5,multipl:[1,5],must:5,mw:[1,2,3],mx:5,my:5,myst:[0,1,2,3,4],mz:5,n:5,name:[1,2,3,5,6],ndarrai:[5,6],need:[1,4],neff:5,neg:5,network:4,neural:4,ngeometri:5,nmultipl:5,nnote:5,none:[1,2,3,5,6,7],nonnegativeint:5,normal:5,normalize_energi:5,normalize_product:5,not_found:[1,2,3],note:[2,4],now:1,np:[1,2,3],num:5,num_cel:[1,2,3],num_mod:[1,2,3,7],num_pml:5,number:[5,7],numpi:[2,4,5,6],nwhich:5,nx2:5,nx:5,ny:5,nz:5,o:4,obj:5,object:[1,5,6,7],odel:4,off:5,offset:2,onc:1,one:5,onli:5,onto:2,oper:5,option:[4,5,7],order:5,orthogon:5,out0:2,out:5,output:2,output_c:2,over:5,overlap:[4,5],oxid:[1,2,3],packag:4,page:4,parallel:[1,2],param:5,paramet:[5,6,7],pars:5,path:[2,5],per:1,perpendicular:[1,2,5],peter:4,phase:5,phd:4,phi:5,photon:4,pip:[1,2,3,4],place:5,plane:[1,2,5],plt:2,plugin:[1,2,3],pm1:3,pm2:3,pml:5,polar:5,poli:[3,5],polygon:[2,5],port:[2,6],port_map:[1,2,3,6],port_nam:2,posit:5,positivefloat:7,positiveint:7,possibl:[1,3],print:[1,2],prism:[3,5],probabl:[2,7],procedur:5,product:5,propag:[1,2,5,6],properti:5,provid:5,px:5,py:[1,2,3,5],pydant:[5,7],pyplot:2,python:[1,2,3,4],pz:5,quick:[0,4],radiu:5,rais:5,rang:[1,2,3],re:1,rectangular:5,ref:5,refer:2,refract:[1,5],region:[1,5],registri:[1,2,3],requir:5,rerun:[1,2,3],result:5,right:[1,2],rigor:4,ring:5,rocm:[1,2,3],round:3,rule:[2,5],s1:3,s2:3,s:[4,6],save:5,sax:[1,4,6],sax_backend:6,scale:[2,4,5],schema:5,scipi:5,search:4,second:1,section:[4,5],sectionw:5,see:[1,4,5],select_port:6,sequenti:1,set:[1,2,3,6],set_sim:7,should:1,show:[1,5],shrink:5,significantli:4,silicon:[1,2,3],silicon_oxid:[1,2,3],sim:[1,2,3,4,7],simpl:[2,4,5],simpler:5,simul:[1,2,4,5,7],singl:5,slab:2,slice:5,small:5,smaller:5,snap:2,snap_to_grid:2,soi:2,solver:[1,4,7],some:2,somewhat:2,sort_structur:5,sourc:[5,6,7],spec:7,specif:[1,3,7],specifi:[1,3,4],speed:4,sphinx:[0,1,2,3,4],srcdoc:[1,2,3],standard:5,start:[0,4,5],stead:4,step:[1,5],str:[5,6],straight:2,string:5,strip:2,struct:2,structur:[4,5],style:[1,2,3],submodul:4,subpackag:4,subset:6,suit:5,suitespars:4,support:5,sy:2,t:[1,2,3,5],t_ox:2,t_slab:2,t_soi:2,take:1,tangenti:5,taper:[0,4],taper_length:2,target_neff:7,te:5,te_fract:5,technic:4,temparatur:1,temperatur:5,tensorflow:[1,2,3],tf_cpp_min_log_level:[1,2,3],than:5,thei:1,therefor:1,thesi:4,theta:5,thi:[1,2,3,4,5],thick:2,those:1,threej:[1,2,3],tidy3d:[4,5,7],time:2,titl:5,tool:4,total:2,toward:4,tpu:[1,2,3],tpu_driv:[1,2,3],tpuplatform:[1,2,3],tqdm:3,transit:2,trimesh:[1,2,3,4],tupl:[2,5],two:[5,6],type:[2,5,7],uid:2,um:7,unabl:[1,2,3],under:1,union:5,uniqu:5,unit:7,up:4,upper:2,us:[1,2,4,5,6],user:2,valid:5,validate_materi:5,validate_typ:5,validationerror:5,valu:5,variabl:5,verlap:4,version:2,vertic:5,vi:[4,5],viewer:[1,2,3],visual:[1,2,3,5],visualize_structur:5,w:4,w_sim:2,wa:[1,5],walk:4,wall:2,want:5,warn:[1,2,3],waveguid:[2,5],wavelength:[1,4,5],we:1,well:1,when:[1,4],wherea:2,which:[4,5,6],width:[1,2,3],width_input:2,width_output:2,width_typ:2,within:5,wl:[1,2,3,5],work:4,worker:[1,2,3],x:[1,2,3,5],x_:5,x_max:[1,5],x_min:[1,5],xla:[1,2,3],xla_bridg:[1,2,3],xla_extens:[1,2,3],xx:5,xy:5,xz:5,y:[1,2,3,5],y_:5,y_max:[1,5],y_min:[1,5],yield:1,you:[1,4],your:[1,4],yx:5,yy:5,yz:5,z:[1,2,5],z_max:[1,5],z_min:[1,5],zero:5,zero_phas:5,zx:[1,2]},titles:["Examples","Quick Start","GDS Taper","Quick Start","meow","meow package","meow.eme package","meow.fde package"],titleterms:{"1":[1,3],"2":[1,3],"3":[1,3],"4":[1,3],"5":[1,3],api:4,avail:4,calcul:[1,2,3],cell:[1,2,3],comput:2,credit:4,cross:[1,2,3],divid:2,doc:4,document:4,em:[1,2,3,6],exampl:[0,2,4],extrud:2,fde:[1,2,3,7],featur:4,find:[1,2,3],full:4,gd:2,home:4,instal:[1,3,4],map:2,matrix:[1,2,3],meow:[4,5,6,7],minim:4,mode:[1,2,3],other:4,packag:[5,6,7],quick:[1,3],s:[1,2,3],section:[1,2,3],select:4,start:[1,3],structur:[1,2,3],submodul:[5,6,7],subpackag:5,taper:2}})