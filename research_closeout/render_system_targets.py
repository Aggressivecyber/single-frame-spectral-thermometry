"""Figure1: frozen synthetic forward and the three prespecified optical targets."""
from . import storage as s
import json
from pathlib import Path


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    selection = s.ROOT/'reports/training_v4/run_001/backbone_selection.json'
    cfg = s.read(selection)['physics_config']
    assert cfg['shape'] == [256, 256]
    assert cfg['response_provenance']['kind'] == 'synthetic_engineering_fixture'
    assert cfg['centers_nm'] == [430, 450, 486, 513, 535, 700, 905]
    spec_path = s.ROOT/'research_closeout/execution_spec.yaml'
    spec = s.read(spec_path)
    sigmas = spec['targets']['extra_Gaussian_sigma_px']
    assert sigmas == [0., 2., 4.]
    folder = s.ROOT/'paper/figures/F01_system_targets'
    folder.mkdir(parents=True, exist_ok=True)
    png, pdf = folder/'SYSTEM_TARGETS_001.png', folder/'SYSTEM_TARGETS_001.pdf'
    assert not png.exists() and not pdf.exists()
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'mathtext.fontset':'dejavusans'})
    fig = plt.figure(figsize=(13.8, 8.0), facecolor='white')
    ax = fig.add_axes((.025, .41, .95, .55)); ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    ax.text(0, 1., '(a) One shared seven-band coded measurement', fontsize=16, weight='bold', va='top')
    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.016',
            facecolor=color,edgecolor='#44546a',linewidth=1.05))
        ax.text(x+w/2,y+h*.70,title,ha='center',va='center',fontsize=12,weight='bold')
        ax.text(x+w/2,y+h*.30,detail,ha='center',va='center',fontsize=11,linespacing=1.5)
    def arrow(start,end):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,lw=1.3,color='#44546a',
                                    connectionstyle='arc3,rad=0'))
    box(.012,.64,.235,.20,'Continuum',r'$\bar C_i=H[C_i(T,a)]$', '#e8f0fa')
    box(.012,.36,.235,.20,'Signed template target',r'$m_i\,u,\quad u=Q\xi$', '#e7f4ef')
    box(.012,.08,.235,.20,'Low-dimensional background',r'$q_i\,H[B(\beta)]$', '#fbf0df')
    box(.305,.30,.19,.32,'Seven band images',r'$I_i=\bar C_i+m_i u$'+'\n'+r'$\quad +q_i H[B(\beta)]$', '#f0f2f5')
    for y in (.74,.46,.18): arrow((.247,y),(.305,.46))
    box(.55,.30,.15,.32,'Spatial encoding',r'$\sum_{i=1}^{7}P_i I_i$', '#e8f0fa')
    arrow((.495,.46),(.55,.46))
    box(.746,.30,.10,.32,'Relay',r'$R[\,\cdot\,]$', '#f0f2f5')
    arrow((.70,.46),(.746,.46))
    box(.902,.30,.082,.32,'Raw', 'noise\nADC', '#fbf0df')
    arrow((.856,.46),(.902,.46))
    ax.text(.40,.19,'430 / 450 / 486 / 513 / 535 / 700 / 905 nm',ha='left',fontsize=11,color='#304966')
    ax.text(.40,.09,'Synthetic response calibration; reference exposure 20 '+r'$\mu$s',ha='left',fontsize=11,color='#304966')
    ax.text(.40,.81,r'$H$: object optical blur     $P_i$: fixed spatial carrier',fontsize=11)
    ax.text(.40,.71,r'$u$ is already postblur; the encoded template has no second $H$',fontsize=11)
    ax.text(.987,.23,r'$256\times256$'+' DN',ha='right',fontsize=10)
    bx = fig.add_axes((.025,.08,.95,.30)); bx.set(xlim=(0,1),ylim=(0,1)); bx.axis('off')
    bx.text(0,1.,'(b) The same targets for every estimator',fontsize=16,weight='bold',va='top')
    for j, sigma in enumerate(sigmas):
        x=.012+j*.333; w=.309
        bx.add_patch(FancyBboxPatch((x,.13),w,.62,boxstyle='round,pad=0.008,rounding_size=0.016',
            facecolor=('#e7f4ef' if sigma==0 else '#eef3f9'),edgecolor='#44546a',lw=1.))
        title='Primary target' if sigma==0 else 'Additional resolution analysis'
        bx.text(x+w/2,.64,title,ha='center',weight='bold',fontsize=12)
        bx.text(x+w/2,.49,rf'$\sigma={sigma:g}$ px',ha='center',fontsize=12)
        bx.text(x+w/2,.33,r'$u_\sigma=G_\sigma*u$',ha='center',fontsize=13)
        bx.text(x+w/2,.19,r'$T_{\mathrm{col},\sigma}=h[(G_\sigma*\bar C_{700})/(G_\sigma*\bar C_{905})]$',ha='center',fontsize=10)
    fig.text(.037,.037,'Template: electrons at 20 '+r'$\mu$s.   Temperature: postblur continuum color temperature in K.   '+
             'Smooth bands before taking their ratio.',fontsize=10.5,color='#304966')
    fig.savefig(png,dpi=180); fig.savefig(pdf); plt.close(fig)
    receipt = {'schema':'system-and-targets-figure-v001','time':s.now(),
        'source':{'path':str(Path(__file__)), 'sha256':s.sha(__file__)},
        'physics_selection':{'path':str(selection),'sha256':s.sha(selection)},
        'spec':{'path':str(spec_path),'sha256':s.sha(spec_path)},
        'centers_nm':cfg['centers_nm'],'shape':cfg['shape'],'reference_exposure_s':cfg['exposure_s'],
        'extra_Gaussian_sigma_px':sigmas,'response_is_synthetic':True,
        'second_template_object_blur_drawn':False,'temperature_map_direct_smoothing_drawn':False,
        'new_inverse_calls':0,'new_raw_generation':0,'final_method_selection_implied':False,
        'outputs':[{'path':str(p),'sha256':s.sha(p),'bytes':p.stat().st_size} for p in (png,pdf)],
        'PNG_visual_QA':'pending','PDF_render_QA':'not_performed'}
    s.write(s.REPORTS/'repair_001/SYSTEM_TARGETS_FIGURE_001.json',receipt)
    print(json.dumps(receipt,indent=2),flush=True)


if __name__ == '__main__':
    run()
