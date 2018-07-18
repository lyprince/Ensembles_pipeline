'''
Author : Luke Prince
Date : 16th July

Neuronal source extraction from motion-corrected video using CNMFE.

Adapted from demo_pipeline_CNMFE by CaImAn team.
'''

import argparse
import yaml
from os import path

import caiman as cm
from caiman.summary_images import correlation_pnr
from caiman.source_extraction import cnmf
from caiman import save_memmap, load_memmap
from caiman.utils.visualization import plot_contours, get_contours
from caiman.components_evaluation import estimate_components_quality_auto


import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap

try:
    import bokeh
    import bokeh.plotting as bpl
    from bokeh.models import CustomJS, ColumnDataSource, Range1d
    from bokeh.io import output_file, show, reset_output
except:
    print("Bokeh could not be loaded. Either it is not installed or you are not running within a notebook")

def get_args():
    parser = argparse.ArgumentParser(description='Neuronal source extracted from motion corrected video using CNMFE')
    parser.add_argument('animal', help='animal ID', nargs=1)
    parser.add_argument('session', help='session: test, trainA, or trainB', nargs=1)
    parser.add_argument('--base_dir', dest = 'base_dir', help='Base directory to find files', default='/home/luke/Documents/Projects/RichardsPostdoc/Ensembles/CA1_imaging/')
    parser.add_argument('-r', '--redo', help='Redo source extraction')
    parser.add_argument('-n', '--n_processes', help='Number of processes', type=int, default=8)
    return parser.parse_args()

if __name__ == '__main__':
    
    def nb_view_patches(Yr, A, C, b, f, d1, d2, YrA=None, image_neurons=None, thr=0.99, denoised_color=None, cmap='jet', save=True, filename='output.html'):
        """
        Interactive plotting utility for ipython notebook
        Parameters:
        -----------
        Yr: np.ndarray
            movie
        A,C,b,f: np.ndarrays
            outputs of matrix factorization algorithm
        d1,d2: floats
            dimensions of movie (x and y)
        YrA:   np.ndarray
            ROI filtered residual as it is given from update_temporal_components
            If not given, then it is computed (K x T)
        image_neurons: np.ndarray
            image to be overlaid to neurons (for instance the average)
        thr: double
            threshold regulating the extent of the displayed patches
        denoised_color: string or None
            color name (e.g. 'red') or hex color code (e.g. '#F0027F')
        cmap: string
            name of colormap (e.g. 'viridis') used to plot image_neurons
        """

        output_file(filename)
        
        colormap = get_cmap(cmap)
        grayp = [mpl.colors.rgb2hex(m) for m in colormap(np.arange(colormap.N))]
        nr, T = C.shape
        nA2 = np.ravel(np.power(A, 2).sum(0)) if type(
            A) == np.ndarray else np.ravel(A.power(2).sum(0))
        b = np.squeeze(b)
        f = np.squeeze(f)
        if YrA is None:
            Y_r = np.array(spdiags(old_div(1, nA2), 0, nr, nr) *
                           (A.T * np.matrix(Yr) -
                            (A.T * np.matrix(b[:, np.newaxis])) * np.matrix(f[np.newaxis]) -
                            A.T.dot(A) * np.matrix(C)) + C)
        else:
            Y_r = C + YrA

        x = np.arange(T)
        if image_neurons is None:
            image_neurons = A.mean(1).reshape((d1, d2), order='F')

        coors = get_contours(A, (d1, d2), thr)
        cc1 = [cor['coordinates'][:, 0] for cor in coors]
        cc2 = [cor['coordinates'][:, 1] for cor in coors]
        c1 = cc1[0]
        c2 = cc2[0]

        # split sources up, such that Bokeh does not warn
        # "ColumnDataSource's columns must be of the same length"
        source = ColumnDataSource(data=dict(x=x, y=Y_r[0] / 100, y2=C[0] / 100))
        source_ = ColumnDataSource(data=dict(z=Y_r / 100, z2=C / 100))
        source2 = ColumnDataSource(data=dict(c1=c1, c2=c2))
        source2_ = ColumnDataSource(data=dict(cc1=cc1, cc2=cc2))

        callback = CustomJS(args=dict(source=source, source_=source_, source2=source2, source2_=source2_), code="""
                var data = source.data
                var data_ = source_.data
                var f = cb_obj.value - 1
                x = data['x']
                y = data['y']
                y2 = data['y2']
                for (i = 0; i < x.length; i++) {
                    y[i] = data_['z'][i+f*x.length]
                    y2[i] = data_['z2'][i+f*x.length]
                }
                var data2_ = source2_.data;
                var data2 = source2.data;
                c1 = data2['c1'];
                c2 = data2['c2'];
                cc1 = data2_['cc1'];
                cc2 = data2_['cc2'];
                for (i = 0; i < c1.length; i++) {
                       c1[i] = cc1[f][i]
                       c2[i] = cc2[f][i]
                }
                source2.change.emit();
                source.change.emit();
            """)

        plot = bpl.figure(plot_width=600, plot_height=300)
        plot.line('x', 'y', source=source, line_width=1, line_alpha=0.6)
        if denoised_color is not None:
            plot.line('x', 'y2', source=source, line_width=1,
                      line_alpha=0.6, color=denoised_color)

        slider = bokeh.models.Slider(start=1, end=Y_r.shape[0], value=1, step=1,
                                     title="Neuron Number", callback=callback)
        xr = Range1d(start=0, end=image_neurons.shape[1])
        yr = Range1d(start=image_neurons.shape[0], end=0)
        plot1 = bpl.figure(x_range=xr, y_range=yr, plot_width=300, plot_height=300)

        plot1.image(image=[image_neurons[::-1, :]], x=0,
                    y=image_neurons.shape[0], dw=d2, dh=d1, palette=grayp)
        plot1.patch('c1', 'c2', alpha=0.6, color='purple',
                    line_width=2, source=source2)

        if Y_r.shape[0] > 1:
            bpl.save(bokeh.layouts.layout([[slider], [bokeh.layouts.row(plot1, plot)]]))
        else:
            bpl.save(bokeh.layouts.row(plot1, plot))
            
        reset_output()
            
        return Y_r

    
    
    def inspect_correlation_pnr(correlation_image_pnr, pnr_image):
        """
        inspect correlation and pnr images to infer the min_corr, min_pnr
        Parameters:
        -----------
        correlation_image_pnr: ndarray
            correlation image created with caiman.summary_images.correlation_pnr
        pnr_image: ndarray
            peak-to-noise image created with caiman.summary_images.correlation_pnr
        Returns:
        -------
        """
        fig = plt.figure(figsize=(10, 4))
        plt.axes([0.05, 0.2, 0.4, 0.7])
        im_cn = plt.imshow(correlation_image_pnr, cmap='viridis')
        plt.title('correlation image')
        plt.colorbar()
        plt.axes([0.5, 0.2, 0.4, 0.7])
        im_pnr = plt.imshow(pnr_image, cmap='viridis')
        plt.title('PNR')
        plt.colorbar()
        
        return fig
    
    args = get_args()
    cell_info = yaml.load(open('./cell_metadata.yaml'))
    
    animal       = args.animal[0]
    session      = args.session[0]
    timestamp    = cell_info[animal][session]['timestamp']
    fileext      = '.mmap'
    
    frame_rate   = cell_info[animal][session]['frame_rate']
    frame_count  = cell_info[animal][session]['frame_count']
    frame_width  = cell_info[animal][session]['frame_width']
    frame_height = cell_info[animal][session]['frame_height']
    
    completed    = cell_info[animal][session]['cnmfe']['completed']
    
    basename     = args.base_dir + '%s/%s_%s_%s'%(animal, timestamp, animal, session)
    
    filename     = basename + '_memmap_d1_%i_d2_%i_d3_1_order_C_frames_%i_%s'%(frame_height, frame_width, frame_count, fileext)
    
    if not completed or args.redo:
        
        Y = Yr, dims, T = load_memmap(filename)
        Y = Yr.T.reshape((T,) + dims, order='F')
        
        # Parameters for source extraction and deconvolution
        parameters = yaml.load(open('./parameters.yaml'))['cnmfe']
        
        p                   = parameters['p']        # Order of autoregressive system
        K                   = parameters['K']        # upper bound on number of components per patch (in general None)
        gSig                = parameters['gSig']     # width of 2D Gaussian kernel, which approximates a neuron
        gSiz                = parameters['gSiz']     # diameter of a CA1 PC (Hippocampus Book), generally gSig*3 + 1
        merge_thresh        = parameters['merge_thresh']  # merging threshold, max correlation allowed
        rf                  = parameters['rf']       # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
        stride              = parameters['stride']   # Overlap between patches. Keep it at least as large as gSiz
        tsub                = parameters['tsub']     # Temporal downsampling factor
        ssub                = parameters['ssub']     # Spatial downsampling factor
        Ain                 = parameters['Ain']      # Initialised components. Can pass as boolean vector if wanted
        low_rank_background = parameters['low_rank_background'] # None leaves background of each patch intact, True performs low rank approximation
        gnb                 = parameters['gnb']      # Number of background components if > 0, elif -2: return BG as b and W, elif -1 return full rank BG as B, elif 0, don't return BG
        nb_patch            = parameters['nb_patch'] # Number of background components per patch
        min_corr            = parameters['min_corr'] # minimum peak value from correlation image
        min_pnr             = parameters['min_pnr']  # minimum peak to noise ratio from PNR image
        ssub_B              = parameters['ssub_B']   # additional spatial downsampling for background
        ring_size_factor    = parameters['ring_size_factor']    # radius of ring is gSiz*ring_size_factor
        
        bord_px             = cell_info[animal][session]['motion_correction']['bord_cnmf']
        
        # compute or retrieve some summary images (correlation and peak to noise)
        
        if path.exists(basename+'_cn_filter.npy') and path.exists(basename+'_pnr.npy'):
            cn_filter = np.load(basename+'_cn_filter.npy')
            pnr = np.load(basename+'_pnr.npy')
            
        else:
            cn_filter, pnr = cm.summary_images.correlation_pnr(Y[::5], gSig=gSig, swap_dim=False)
            np.save(basename+'_cn_filter.npy', cn_filter)
            np.save(basename+'_pnr.npy', pnr)

            fig = inspect_correlation_pnr(cn_filter, pnr)
            fig.savefig(args.base_dir+'%s/%s_%s_%s_corr_pnr_image.svg'%(animal, timestamp, animal, session))
            plt.close()
        
#         import pdb
#         pdb.set_trace()
        
        cnm = cnmf.CNMF(n_processes=args.n_processes, method_init='corr_pnr', k=K,
                gSig=(gSig, gSig), gSiz=(gSiz, gSiz),
                merge_thresh = merge_thresh, p=p, dview= None, #dview,
                tsub=tsub, ssub=ssub, Ain=Ain, rf=rf, stride= stride,
                only_init_patch=True, gnb=gnb, nb_patch=nb_patch, method_deconvolution='oasis',
                low_rank_background=low_rank_background, update_background_components=True,
                min_corr=min_corr, min_pnr=min_pnr, normalize_init=False, center_psf = True,
                ssub_B=ssub_B, ring_size_factor = ring_size_factor, del_duplicates=True, border_pix=bord_px)
        
        cnm.fit(Y)
        
        crd = plot_contours(cnm.A, cn_filter, thr=.8, vmax=0.99)
        
        # Parameters for component evaluation
        parameters   = yaml.load(open('./parameters.yaml'))['component_evaluation']
        decay_time   = parameters['decay_time']
        min_SNR      = parameters['min_SNR']
        r_values_min = parameters['r_values_min']
        
        idx_components, idx_components_bad, comp_SNR, r_values, pred_CNN = estimate_components_quality_auto(
                            Y, cnm.A, cnm.C, cnm.b, cnm.f, cnm.YrA, frame_rate, 
                            decay_time, gSig, dims, dview=None, 
                            min_SNR=min_SNR, r_values_min=r_values_min, use_cnn=False)
        
        fig = plt.figure(figsize=(15,8));
        plt.subplot(121);
        crd = plot_contours(cnm.A.tocsc()[:,idx_components], cn_filter, thr=.8, vmax=0.95)
        plt.title('Contour plots of accepted components')
        plt.subplot(122); 
        crd = plot_contours(cnm.A.tocsc()[:,idx_components_bad], cn_filter, thr=.8, vmax=0.95)
        plt.title('Contour plots of rejected components')
        fig.savefig(basename+'_cnmfe_components_spatial.svg')
        plt.close()
        
        # Accepted Components
        nb_view_patches(Yr, cnm.A.tocsc()[:, idx_components], cnm.C[idx_components], 
                cnm.b, cnm.f, dims[0], dims[1], YrA=cnm.YrA[idx_components], image_neurons=cn_filter,
                denoised_color='red', thr=0.8, cmap='gray', save=True, filename=basename+'_cnmfe_components_accepted.html')
        
        # Rejected Components
        nb_view_patches(Yr, cnm.A.tocsc()[:, idx_components_bad], cnm.C[idx_components_bad], 
                cnm.b, cnm.f, dims[0], dims[1], YrA=cnm.YrA[idx_components_bad], image_neurons=cn_filter,
                denoised_color='red', thr=0.8, cmap='gray', save=True, filename=basename+'_cnmfe_components_rejected.html');
        
        nrn_movie = np.reshape(cnm.A.tocsc()[:,idx_components].dot(cnm.C[idx_components]),dims+(-1,), order = 'F').transpose(2,0,1)
        
        save_memmap([nrn_movie], base_name = basename + '_neurons_memmap', order= 'F', border_to_0 = bord_px) 
