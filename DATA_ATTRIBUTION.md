# Experimental image sources

The model weights were trained on synthetic fixtures. The expanded shape tests additionally use observed image morphology from these experimental datasets:

| Dataset | Creators | Source | License |
|---|---|---|---|
| Flat premixed CH4/air standard flames, version 6 | Tanja Pelzmann; Etienne Robert | [10.17632/jvvvgpdy48.6](https://data.mendeley.com/datasets/jvvvgpdy48/6) | CC BY 4.0 |
| Urea/liquid-fuel low-NOx combustor, version 3 | Maria Grazia De Giorgi; Giuseppe Ciccarella; Donato Fontanarosa; Elisa Pescini; Antonio Ficarella | [10.17632/2tjc68s5vf.3](https://data.mendeley.com/datasets/2tjc68s5vf/3) | CC BY 4.0 |
| LEAFINNOX_Data | Epaminondas Mastorakos; Pedro M de Oliveira | [10.17863/CAM.90803](https://www.repository.cam.ac.uk/items/1d2e878a-d73a-4cc3-880c-fd639c883ede) | CC BY 4.0 |

Derived image panels and processed experimental shape arrays retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The MIT license for our code and synthetic model weights does not replace these source licenses. Original archives are not redistributed here.

Adaptations: fixed cropping to remove camera text and edge artifacts; masking source saturation; area averaging while preserving aspect ratio; contrast normalization from the fifth to the 99.5th percentile; zero letterboxing to 256×256. Metadata records the native file/member, frame index, hashes, native bit depth, crop and masked fraction. PNG RGB values are used only as a spatial intensity descriptor; no radiometric calibration is inferred.

For the shape-transfer experiment, processed CH*/OH* images define a spatial coefficient pattern for the existing synthetic active template. Independently recorded broadband/soot shapes define a constructed background. Neither operation establishes measured AlO, synchronized channel pairs or local temperature truth. Temperatures, spectral response, coefficient amplitude, spatial coding and camera noise are simulated. These experiments assess reconstruction of real-image-shaped patterns, with quantitative error against the constructed target.
