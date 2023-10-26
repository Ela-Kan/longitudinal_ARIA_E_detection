"""
Class to perform the Z-score and clustering analysis on the 3D maps generated for the variance and gradient
of the ARIA-E longitudinal data.

Author: Ela Kanani


"""

import numpy as np
import os
import pandas as pd
import shutil
import nipype.interfaces.fsl as fsl
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib

class ZscoreClustering():

    def __init__(self) -> None:
        """
        Initialises instance of the class.
        """
        pass

    def applyMask(self, input_file, output_file, mask_file):
        """
        Given the path to a brain mask, applies it to the input file using FSL.
        This can be a brain extraction mask, or a mask of the region of interest.
    
        Parameters
        ----------

        input_file: str
            input file to be masked

        output_file: str
            output file name

        mask_file: str
             Mask file name

         Returns
        -------
        None
        """

        apply_mask = fsl.ApplyMask()
        apply_mask.inputs.in_file = input_file
        apply_mask.inputs.out_file = output_file
        apply_mask.inputs.mask_file = mask_file
        apply_mask.run()

        print("Mask successfully applied.")

        return None
    
    def clusterZscore(self, Z_score_map_file, cluster_info_out_file, cluster_out_file, threshold):
        """
        Given a Z-score map, clusters the voxels based on the threshold provided using FSL. Here, 26 connectivity is used.

        Parameters
        ----------
        Z_score_map_file: str
            Z-score map file name

        cluster_info_out_file: str
            Cluster information output file name for text info (i.e. voxel size, ID, etc)
        
        cluster_out_file: str
            Cluster output file name for the clustered map

        threshold: float
            Significance threshold to be used for clustering

        Returns
        -------
        None
        """
        
        # FSL command
        command = f"fsl-cluster -i {Z_score_map_file} -t {threshold} -o {cluster_out_file} -> {cluster_info_out_file}"
        # run command
        os.system(command)
        
        print("Z-score map clustered.")
        return None
    
    def filterClusters(self, cluster_map_file, cluster_info_txt_file, thresholded_cluster_map_file, adjusted_intensity_thresholded_cluster_map_file = []):
        """
        Sifts through cluster information in the text file and removes clusters smaller than 10 (determined as the noise threshold).

        Parameters
        ----------
        cluster_map_file: str
            Cluster map file name
        
        cluster_info_txt_file: str
            Cluster information text file name

        thresholded_cluster_map_file): str
            Thresholded cluster map file name for output

        adjusted_intensity_thresholded_cluster_map_file: str
            Thresholded cluster map file name for output with adjusted intensity values (i.e. intensity values are the cluster size). This is optional.

        Returns
        -------
        None
        """
        
        # read in cluster info as a dataframe
        cluster_info = pd.read_csv(cluster_info_txt_file, delimiter='\t') 

        # extract the clusters with more than or equal to 10 voxels
        big_clusters = cluster_info[cluster_info['Voxels'] >= 10]

        # threshold the cluster map to only include the big clusters
        threshold = fsl.Threshold()
        threshold.inputs.in_file = cluster_map_file
        threshold.inputs.thresh = np.min(big_clusters['Cluster Index'].tolist())
        threshold.inputs.direction = 'below'
        threshold.inputs.out_file = thresholded_cluster_map_file
        threshold.run()
       
        print("Clusters filtered.")

        # if the adjusted intensity thresholded cluster map file name is provided
        if len(adjusted_intensity_thresholded_cluster_map_file):
            thresholded_cluster_map = nib.load(thresholded_cluster_map_file)
            unique_labels = np.unique(thresholded_cluster_map.get_fdata())
            # for each cluster, replace the intensity value with the cluster size
            voxel_size = big_clusters['Voxels'].tolist()
            voxel_size.reverse() # we want the sizes to ascend
            voxel_size.insert(0,0) # for the background

            # extract the vozels from big clusters
            value_to_intensity = {original:vox_size for original, vox_size in zip(unique_labels, voxel_size)}
            vox_mapped_image = np.vectorize(value_to_intensity.get, otypes=[int])(thresholded_cluster_map.get_fdata())

            # save the image
            mapped_vox_size_var_no_csf_nifti =  nib.Nifti1Image(vox_mapped_image, affine=None)
            # ensure that the coordinate frame is correct (i.e. match the thresholded cluster map)
            mapped_vox_size_var_no_csf_nifti.set_sform(thresholded_cluster_map.get_sform())
            mapped_vox_size_var_no_csf_nifti.set_qform(thresholded_cluster_map.get_qform())
            nib.save(mapped_vox_size_var_no_csf_nifti, adjusted_intensity_thresholded_cluster_map_file)

            print("Adjusted intensity thresholded cluster map saved.")

        return None
    

    def separateClusteredissueCSF(self, cluster_map_file, CSF_mask_in_file, tissue_cluster_map_out, csf_cluster_map_out):
        """
        Given a cluster map, separate out the tissue and csf clusters. This could be used to mask out noise in lower parts
        of the brain.
        
        Parameters
        ----------
        cluster_map_file: str
            Cluster map file name

        CSF_mask_in_file: str
            CSF mask file name
        
        tissue_cluster_map_out: str
            Tissue cluster map file name

        csf_cluster_map_out: str
            CSF cluster map file name
        """

        # load in the csf mask
        csf_mask = nib.load(CSF_mask_in_file)
        # load in the cluster map
        cluster_map = nib.load(cluster_map_file)

        # extract the csf clusters
        csf_cluster_map = np.multiply(csf_mask.get_fdata(), cluster_map.get_fdata())
        # extract the tissue clusters
        inverse_csf_mask = np.logical_not(csf_mask.get_fdata()) * 1 #* 1 converts to 1 or 0 from a true/false to allow multiplication
        cluster_no_csf = np.multiply(inverse_csf_mask, cluster_map.get_fdata())

        # save the maps, ensuring that the coordinate frame is correct
        csf_cluster_map_nifti =  nib.Nifti1Image(csf_cluster_map, affine=None)
        csf_cluster_map_nifti.set_sform(cluster_map.get_sform())
        csf_cluster_map_nifti.set_qform(cluster_map.get_qform())
        nib.save(csf_cluster_map_nifti, csf_cluster_map_out)

        cluster_no_csf_nifti =  nib.Nifti1Image(cluster_no_csf, affine=None)
        cluster_no_csf_nifti.set_sform(cluster_map.get_sform())
        cluster_no_csf_nifti.set_qform(cluster_map.get_qform())
        nib.save(cluster_no_csf_nifti, tissue_cluster_map_out)

        print("CSF and tissue cluster maps saved.")

        return None





    