import numpy as np  
import math  
import os  
import scipy.io as scio  
import ctypes  
import time   
import glob   
import scipy  
import argparse     
from sklearn.metrics import mean_squared_error        
from path_planning.kmp.demo_GMR import *      
from path_planning.utils_functions import *    
from path_planning.spm_path.spm_kinematics import *    

from matplotlib.pyplot import title    
import seaborn as sns     

 
# sns.set(font_scale=1.5)     
np.set_printoptions(precision=4)      



def generate_ori_path(args):      
    spm_fik_left = SPM_FIK(beta=60, alpha_1=45, alpha_2=45)   
    spm_fik_right = SPM_FIK(beta=60, alpha_1=45, alpha_2=45)   
             
    # r = spm_fik.R      
    # r_1 = spm_fik.R_1     
     
    # sample_num_1=[1, 5000]    
    # sample_num_1=[1, 100]      
    # sample_num_1=[2000, 1]       
    sample_num_1=[1, 4000]        
    # sample_num_1=[5000, 1]        
    # sample_num_1=[2000, 1]     
       
    sample_num_2=50         
    
    # theta_range=[7/8*np.pi,9/8*np.pi]   
    # theta_range=[8/9*np.pi,10/9*np.pi]   
    # theta_range=[np.pi,7/8*np.pi]    
    # theta_range=[np.pi,9/8*np.pi]     
    # theta_range=[0.0, 0.0]  
    # theta_range=[np.pi,10/9*np.pi]     
    # theta_range=[np.pi,np.pi]  
    # theta_range=[7/8 * np.pi, 9/8*np.pi]       
    # theta_range=[7/8*np.pi, 9/8*np.pi]      
    # theta_range=[8/9 * np.pi, 8/9*np.pi]     
    # theta_range=[6/8 * np.pi, 10/8*np.pi]    
    # theta_range=[9/8*np.pi,9/8*np.pi]  
    theta_range=[10/9*np.pi,10/9*np.pi]  
    
    # phi_range=[0,0]   
    # phi_range=[np.pi*3/4,np.pi*3/4]   
    # phi_range=[np.pi,np.pi]     
    # phi_range=[3/2*np.pi,3/2*np.pi]   
    # phi_range=[2*np.pi,2*np.pi]
    # phi_range=[0,1/2*np.pi]   
    # phi_range=[0,np.pi]   
    # phi_range=[0,2*np.pi]  
    phi_range=[2*np.pi,0]  
    # phi_range=[np.pi/4.0,np.pi/4.0]      
    # phi_range=[-np.pi/2,-np.pi/2]      
    # phi_range=[-np.pi,-np.pi]      
    # phi_range=[np.pi/2,np.pi/2]      
    # phi_range=[np.pi*3.0/4.0,np.pi*3.0/4.0]       
    
    # psi=-54.73561     
    psi=0.0     
    plot_index=0      

    trajectory_theta_list, forward_n_list, true_n_list, theta_t_list, v_list_all, center_point = generate_sphere_reference(
        [0, 0, 0],    
        spm_fik_left,    
        sample_num_1=sample_num_1,       
        sample_num_2=sample_num_2,     
        index=plot_index,     
        theta_range=theta_range,       
        phi_range=phi_range,     
        psi=psi    
    )   
    
    final_list = np.hstack((trajectory_theta_list,center_point))
    print("center_point :", final_list.shape)                  
    
    np.savetxt("./data/wrist_demo/demo_data/trajectory_theta_list_" + args.flag + ".txt", final_list, fmt='%f', delimiter=',')    
    
    
    # plot_single_spm_path(
    #     spm_fik=spm_fik_left,    
    #     center_point=center_point,     
    #     v_list=v_list_all,     
    #     sample_num_2=sample_num_2,     
    #     plot_index=plot_index,     
    #     save_fig=True       
    # )  



if __name__ == "__main__":   
    parser = argparse.ArgumentParser()     

    parser.add_argument('--mode', type=int, default=0, help='choose mode first !!!!')       
    parser.add_argument('--ctl_mode', type=str, default="zero_force", help='choose mode first !!!!')       
    # //// basics ///////////  
    parser.add_argument('--num', type=int, default=3000, help='choose index first !!!!')       
    parser.add_argument('--iter', type=int, default=1, help='choose index first !!!!')       
    parser.add_argument('--ee_force', type=int, default=0, help='select from {1, 2, 3}')       
    parser.add_argument('--speed', type=int, default=500, help='select from {1, 2, 3}')      
    parser.add_argument('--delay_time', type=float, default=5.0, help='select from {1, 2, 3}')      
    # //// game /////////////
    parser.add_argument('--bilateral', type=int, default=0, help='choose mode first !!!!')       
    parser.add_argument('--use_vr', type=int, default=0, help='choose mode first !!!!')       
    parser.add_argument('--game_index', type=int, default=1, help='select from {1, 2, 3}')        
    parser.add_argument('--motion_mode', type=int, default=0, help='select from {1, 2, 3}')        
    parser.add_argument('--target_index', type=int, default=1, help='select from {0, ..., 7}')        
    parser.add_argument('--target_random', type=int, default=0, help='select from {0, 1}')        
    # //// path /////////////
    parser.add_argument('--flag', type=str, default='fifth', help='choose index first !!!!')      
    parser.add_argument('--file_name', type=str, default='x_p', help='load reference trajectory !!!')       
    parser.add_argument('--root_path', type=str, default='./data/wrist_demo', help='choose index first !!!!')     
    # //// learning /////////
    parser.add_argument('--nb_data', type=int, default=200, help='choose index first !!!!')      
    parser.add_argument('--nb_samples', type=int, default=5, help='load reference trajectory !!!')       
    parser.add_argument('--nb_states', type=int, default=10, help='choose index first !!!!')      
    parser.add_argument('--data_name', type=str, default="iteration_learning", help='choose index first !!!!')      
    parser.add_argument('--sample_num', type=int, default=5000, help='choose index first !!!!')      


    # # # //////////////////////////////////////////////////////////
    # spm_fik_left = SPM_FIK(beta=60, alpha_1=45, alpha_2=45)    

    # # # # print(n_1) 
    # # # # print(n_2) 
    # # # # print(n_3)      
    
    # # for i in range(4000):  
    # #     center_point = np.array(n_list[i, :])  
    # #     psi = 0
    # #     point_scatter, n_1, n_2, n_3 = get_3D_point(
    # #         r=spm_fik_left.R_1,  
    # #         center=center_point,     
    # #         n=center_point,    
    # #         sample=np.array([0.0+psi, 120.0+psi, 240.0+psi])    
    # #     )   
    # #     print("center_point :", center_point, n_1)  
    # #     deform_q = spm_fik_left.cal_IK(
    # #         v_t_1=n_1,   
    # #         v_t_2=n_2,   
    # #         v_t_3=n_3     
    # #     )   
        
    # #     real_list[i, :] = deform_q    
    
    # for i in range(N):  
    #     print("center :", np.array(real_list_n[start_index+i:start_index+i+1, :]))
    #     center_point = np.array(real_list_n[start_index+i:start_index+i+1, :][0])  
    #     _, n_1, n_2, n_3 = get_3D_point(
    #         r=spm_fik_left.R_1,  
    #         center=center_point,     
    #         n=center_point,    
    #         sample=np.array([0.0,120.0,240.0])    
    #     )   
        
    #     # _, n_1, n_2, n_3 = get_3D_point(
    #     #     r=spm_fik_left.R_1,  
    #     #     center=np.array(real_list[start_index+i:start_index+i+1, :][0]),     
    #     #     n=np.array(real_list[start_index+i:start_index+i+1, :][0]),    
    #     #     sample=np.array([0.0,120.0,240.0])    
    #     # )    
        
    #     deform_q = spm_fik_left.cal_IK(
    #         v_t_1=n_1,   
    #         v_t_2=n_2,   
    #         v_t_3=n_3     
    #     )   
    #     print("deform_q", deform_q)    
        
    #     real_list[start_index+i:start_index + i+1, :] = deform_q    