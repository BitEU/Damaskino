/*
 * Program Name: Damaskino
 * Program Release Year: 2025
 * Program Author: Steven S.
 * Program Link: https://github.com/BitEU/Damaskino
 * Purpose: Simulate nuclear fallout dispersion and radiation levels using the WSEG-10 model
 */

/*
 * WSEG-10 NUCLEAR FALLOUT PREDICTION MODEL
 * Ultra-detailed implementation for civil defense planning
 * 
 * PHYSICS MODEL: WSEG-10 (Weapons Systems Evaluation Group Report #10, 1959)
 * 
 * PARTICLE PHYSICS:
 * - Log-normal particle size distribution (10 μm to 2000 μm)
 * - Median diameter: 100-150 μm, σ_g = 2.0-3.0
 * - Stokes settling: v = g*d²*(ρ_p - ρ_a)/(18*μ)
 * - Transition to Newton drag for d > 500 μm
 * 
 * CLOUD DYNAMICS:
 * - Mushroom cloud rise: H_top = 12.5*W^0.25 km (W in KT)
 * - Stabilization time: t_stab = 10*W^0.25 minutes
 * - Cloud stem diameter: D_stem = 1.5*W^0.33 km
 * 
 * RADIOACTIVE DECAY:
 * - Way-Wigner formula: I(t) = I₀*t^(-1.2)
 * - Reference: 1 MT at 1 hour = 1.6×10⁶ R/hr at 1 NM
 * - Fractionation by particle size (refractory vs volatile)
 * 
 * WIND TRANSPORT:
 * - Multi-layer atmospheric model (0, 5, 10, 20, 30, 50 Kft)
 * - Wind shear and directional variation
 * - Turbulent diffusion (σ_y, σ_z growth with distance)
 * 
 * DEPOSITION MODEL:
 * - Gravitational settling with eddy diffusion
 * - Surface roughness and deposition velocity
 * - Hot particle clustering (local enhancement)
 * 
 * TELETYPE COMPATIBLE:
 * - 72-column output format
 * - ASCII art visualization
 * - Tabular data presentation
 * 
 * Reference: WSEG Report #10 (1959), Glasstone & Dolan (1977)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include "terrain_data.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Grid configuration
#define GRID_SIZE 64                  // 64x64 grid (reduced to shrink memory/buffers)
#define GRID_RESOLUTION_KM 2.0f       // 2 km per cell
#define MAX_DOWNWIND_KM 160.0f        // 160 km maximum range

// Terrain configuration
#define MAX_TERRAIN_DIM 50
#define ELEV_SCALE 306.67f            // Convert 4-bit (0-15) to meters: 4600/15
#define ELEV_OFFSET -100.0f

// Particle size classes (WSEG-10 distribution)
#define NUM_PARTICLE_CLASSES 12
static const float PARTICLE_DIAMETERS[NUM_PARTICLE_CLASSES] = {
    10, 20, 40, 70, 100, 150, 250, 400, 600, 900, 1300, 2000  // microns
};

// Wind altitude layers (feet)
#define NUM_WIND_LAYERS 6
static const float WIND_ALTITUDES_FT[NUM_WIND_LAYERS] = {
    0, 5000, 10000, 20000, 30000, 50000
};

// Physics constants
#define GRAVITY 9.81f                 // m/s²
#define AIR_DENSITY 1.225f            // kg/m³ at sea level
#define AIR_VISCOSITY 1.81e-5f        // Pa·s at 15°C
#define PARTICLE_DENSITY 2500.0f      // kg/m³ (silicate debris)
#define KNOTS_TO_MS 0.514444f         // conversion factor

// Radiation constants (WSEG-10)
#define REF_ACTIVITY_1MT_1HR 1.6e6f   // R/hr at 1 NM from 1 MT at H+1
#define NAUTICAL_MILE_M 1852.0f
#define DECAY_EXPONENT 1.2f           // Way-Wigner decay

// Data structures
typedef struct {
    float speed_kts;                 // Wind speed in knots
    float direction_deg;             // Wind direction (from, meteorological)
} WindLayer;

typedef struct {
    float yield_kt;                  // Weapon yield in kilotons
    int is_surface_burst;             // 1 = surface, 0 = air burst
    float hob_m;                     // Height of burst in meters
    float fission_fraction;          // Fraction of yield from fission
} WeaponParams;

typedef struct {
    WindLayer layers[NUM_WIND_LAYERS];
    float shear_factor;              // Wind shear coefficient
} AtmosphereModel;

typedef struct {
    float cloud_top_km;              // Mushroom cloud top altitude
    float cloud_base_km;             // Cloud base altitude
    float stem_diameter_km;          // Stem diameter
    float stabilization_time_min;    // Time to stabilize
} CloudModel;

typedef struct {
    float median_microns;            // Median particle diameter
    float geometric_sigma;           // Log-normal σ_g
    float mass_fractions[NUM_PARTICLE_CLASSES];  // Mass distribution
} ParticleDistribution;

typedef struct {
    float dose_rate_rhr;             // Dose rate in R/hr at H+1
    float arrival_time_hr;           // Time of first arrival
    int particle_class_max;           // Dominant particle size class
} GridCell;

typedef struct {
    unsigned char data[MAX_TERRAIN_DIM*MAX_TERRAIN_DIM];
    int w,h;
    int active;
} TerrainData;

typedef struct {
    WeaponParams weapon;
    AtmosphereModel atmosphere;
    CloudModel cloud;
    ParticleDistribution particles;
    GridCell grid[GRID_SIZE][GRID_SIZE];
    int gz_x, gz_y;                   // Ground zero coordinates
    float time_since_burst_hr;       // Reference time for dose rate
    TerrainData terrain;
} FalloutModel;

// Location names for terrain tiles
#define NUM_LOCATIONS 18
// Location names for terrain tiles
#define NUM_LOCATIONS 18
static const char* LOCATION_NAMES[NUM_LOCATIONS+1] = {
    "FLAT (NO TERRAIN)",
    "Los Angeles",
    "San Francisco",
    "NORAD Peterson SFB",
    "Washington D.C.",
    "Chicago",
    "Barksdale AFB",
    "Whiteman AFB",
    "Strategic Command, Offutt AFB",
    "Jamesburg 08831",
    "Joint Base McGuire-Dix-Lakehurst",
    "Camp Evans Wall Township",
    "Kirtland AFB Albuquerque",
    "New York",
    "Philadelphia",
    "Raven Rock Mountain Complex",
    "Saint Marys 15857 Summa Facility",
    "Naval Station Norfolk",
    "Naval Base Kitsap",
};



// Function prototypes
void print_teletype_banner(void);
void initialize_model(FalloutModel* model);
int select_location(FalloutModel* model);
int load_terrain(FalloutModel* model, int loc);
float get_terrain_elev(FalloutModel* model, float x_km, float y_km);
void input_weapon_parameters(FalloutModel* model);
void input_wind_profile(FalloutModel* model);
void calculate_cloud_parameters(FalloutModel* model);
void calculate_particle_distribution(FalloutModel* model);
void calculate_fallout_pattern(FalloutModel* model);
float get_settling_velocity(float diameter_microns, float altitude_m);
float get_fission_activity(float yield_kt, float time_hr, float fission_fraction);
void calculate_transport_and_deposition(FalloutModel* model, int particle_class);
float gaussian_plume(float x, float y, float sigma_x, float sigma_y);
void display_results(FalloutModel* model);
void display_dose_map(FalloutModel* model);
void display_contours(FalloutModel* model);
void display_sector_report(FalloutModel* model);
void display_timeline(FalloutModel* model);
char* trim_input(char* str);
int parse_wind_input(const char* input, float* speed, float* direction);

// Main program
int main(void) {
    FalloutModel model;
    char response[80];
    
    print_teletype_banner();
    
    while (1) {
        printf("\n");
        printf("========================================================\n");
        printf("                 NEW SIMULATION SESSION\n");
        printf("========================================================\n");
        
        initialize_model(&model);
        if(!select_location(&model))continue;
        input_weapon_parameters(&model);
        input_wind_profile(&model);
        
        printf("\n*** CALCULATING CLOUD DYNAMICS ***\n");
        calculate_cloud_parameters(&model);
        
        printf("*** COMPUTING PARTICLE DISTRIBUTION ***\n");
        calculate_particle_distribution(&model);
        
        printf("*** SIMULATING FALLOUT DEPOSITION ***\n");
        printf("(This may take a moment...)\n");
        calculate_fallout_pattern(&model);
        
        printf("*** COMPUTATION COMPLETE ***\n");
        
        display_results(&model);
        
        printf("\n\nRUN ANOTHER SIMULATION? (Y/N): ");
        if (fgets(response, sizeof(response), stdin)) {
            char first = toupper(response[0]);
            if (first != 'Y') {
                break;
            }
        } else {
            break;
        }
    }
    
    printf("\n========================================================\n");
    printf("           SIMULATION SESSION TERMINATED\n");
    printf("========================================================\n");
    
    return 0;
}

// Print teletype-style banner
void print_teletype_banner(void) {
    printf("\n\n");
    printf("========================================================\n");
    printf("    WSEG-10 NUCLEAR FALLOUT PREDICTION CALCULATOR\n");
    printf("========================================================\n");
    printf("\n");
    printf("WEAPONS SYSTEMS EVALUATION GROUP - REPORT NO. 10\n");
    printf("DETAILED FALLOUT PATTERN PREDICTION MODEL\n");
    printf("\n");
    printf("PHYSICS IMPLEMENTATION:\n");
    printf("  - Log-normal particle size distribution\n");
    printf("  - Stokes/Newton settling dynamics\n");
    printf("  - Multi-layer atmospheric wind transport\n");
    printf("  - Way-Wigner radioactive decay (t^-1.2)\n");
    printf("  - Gaussian plume diffusion model\n");
    printf("\n");
    printf("OPTIMIZED FOR TELETYPE OUTPUT (72 COLUMNS)\n");
    printf("\n");
    printf("Enter 'QUIT' at any prompt to exit.\n");
    printf("\n");
}

// Initialize model with defaults
void initialize_model(FalloutModel* model) {
    memset(model, 0, sizeof(FalloutModel));
    
    // Ground zero at center
    model->gz_x = GRID_SIZE / 2;
    model->gz_y = GRID_SIZE / 2;
    
    // Reference time H+1
    model->time_since_burst_hr = 1.0;
    
    // Default atmosphere (calm)
    for (int i = 0; i < NUM_WIND_LAYERS; i++) {
        model->atmosphere.layers[i].speed_kts = 0.0;
        model->atmosphere.layers[i].direction_deg = 270.0;  // From west
    }
    model->atmosphere.shear_factor = 1.0;
    model->terrain.active = 0;
}

// Select target location and load terrain
int select_location(FalloutModel* model) {
    char input[80];
    printf("\n*** TARGET LOCATION ***\n\n");
    printf(" 0: FLAT (no terrain)\n");
    for(int i=1;i<=NUM_LOCATIONS;i++) printf("%2d: %s\n",i,LOCATION_NAMES[i]);
    printf("\nSELECT LOCATION (0-%d): ", NUM_LOCATIONS);
    if(!fgets(input,sizeof(input),stdin))exit(0);
    if(strcmp(trim_input(input),"QUIT")==0)exit(0);
    int loc;
    if(sscanf(input,"%d",&loc)!=1||loc<0||loc>NUM_LOCATIONS){
        printf("ERROR: Enter 0-%d.\n", NUM_LOCATIONS);return 0;
    }
    if(loc==0){model->terrain.active=0;printf("\nUsing flat terrain model.\n");}
    else if(!load_terrain(model,loc)){
        printf("ERROR: Cannot load terrain for %s\n",LOCATION_NAMES[loc]);return 0;
    }else printf("\nLoaded terrain: %s (%dx%d)\n",LOCATION_NAMES[loc],model->terrain.w,model->terrain.h);
    return 1;
}

// Load terrain from embedded data (4-bit delta encoded)
int load_terrain(FalloutModel* model, int loc) {
    if(loc<1||loc>NUM_LOCATIONS) return 0;
    int off=TERRAIN_INDEX[loc-1][0],len=TERRAIN_INDEX[loc-1][1];
    if(len<6)return 0;
    const unsigned char*p=TERRAIN_BLOB+off;
    model->terrain.w=p[0]|(p[1]<<8);
    model->terrain.h=p[2]|(p[3]<<8);
    int datalen=p[4]|(p[5]<<8);
    p+=6;
    // Decode 4-bit delta: first byte has first_val in high nibble
    int idx=0,val=(p[0]>>4)&0x0F;
    model->terrain.data[idx++]=val;
    // Unpack remaining nibbles as deltas
    int nibble=1;  // Start from low nibble of first byte
    for(int i=0;i<datalen&&idx<MAX_TERRAIN_DIM*MAX_TERRAIN_DIM;){
        int d;
        if(nibble&1)d=p[i++]&0x0F;
        else d=(p[i]>>4)&0x0F;
        nibble++;
        val+=d-8;  // Delta is stored as 0-15, subtract 8 for -7..+7
        if(val<0)val=0;if(val>15)val=15;
        model->terrain.data[idx++]=val;
    }
    model->terrain.active=1;
    return 1;
}

// Get terrain elevation at position (km from GZ)
float get_terrain_elev(FalloutModel* model, float x_km, float y_km) {
    if(!model->terrain.active)return 0.0;
    // Map km to terrain grid (terrain covers ~200km, centered)
    float scale = 200.0f / model->terrain.w;
    int tx=(int)((x_km+100.0)/scale);
    int ty=(int)((y_km+100.0)/scale);
    if(tx<0||tx>=model->terrain.w||ty<0||ty>=model->terrain.h)return 0.0;
    return model->terrain.data[ty*model->terrain.w+tx]*ELEV_SCALE+ELEV_OFFSET;
}

// Input weapon parameters
void input_weapon_parameters(FalloutModel* model) {
    char input[80];
    
    printf("\n*** WEAPON PARAMETERS ***\n\n");
    
    // Yield
    while (1) {
        printf("ENTER WEAPON YIELD (KT): ");
        if (!fgets(input, sizeof(input), stdin)) exit(0);
        if (strcmp(trim_input(input), "QUIT") == 0) exit(0);
        
        float yield;
        if (sscanf(input, "%f", &yield) == 1 && yield > 0 && yield <= 100000.0f) {
            model->weapon.yield_kt = yield;
            break;
        }
        printf("ERROR: Enter yield between 0 and 100000 KT.\n");
    }
    
    // Burst type
    while (1) {
        printf("BURST TYPE (SURFACE/AIR): ");
        if (!fgets(input, sizeof(input), stdin)) exit(0);
        char* trimmed = trim_input(input);
        if (strcmp(trimmed, "QUIT") == 0) exit(0);
        
        if (strstr(trimmed, "SURFACE") || strstr(trimmed, "GROUND") || strcmp(trimmed, "S") == 0) {
            model->weapon.is_surface_burst = 1;
            model->weapon.hob_m = 0.0f;
            model->weapon.fission_fraction = 1.0f;  // Assume pure fission for simplicity
            break;
        } else if (strstr(trimmed, "AIR") || strcmp(trimmed, "A") == 0) {
            model->weapon.is_surface_burst = 0;
            // Optimal HOB for blast: 540 * Y^0.4 meters
            model->weapon.hob_m = 540.0f * powf(model->weapon.yield_kt / 1000.0f, 0.4f);
            model->weapon.fission_fraction = 0.5;  // Typical for fusion weapons
            printf("  (Height of burst set to %.0f meters)\n", model->weapon.hob_m);
            printf("  (WARNING: Airburst produces minimal local fallout)\n");
            break;
        }
        printf("ERROR: Enter SURFACE or AIR.\n");
    }
    
    printf("\nWEAPON CONFIGURATION:\n");
    printf("  Yield: %.1f KT\n", model->weapon.yield_kt);
    printf("  Type: %s\n", model->weapon.is_surface_burst ? "SURFACE BURST" : "AIR BURST");
    printf("  Fission fraction: %.0f%%\n", model->weapon.fission_fraction * 100);
}

// Input wind profile
void input_wind_profile(FalloutModel* model) {
    char input[80];
    
    printf("\n*** WIND PROFILE ***\n\n");
    printf("Enter wind as: SPEED@DIRECTION (e.g., 15@270)\n");
    printf("Direction is meteorological (wind FROM this direction)\n");
    printf("You can specify winds at different altitudes, or use\n");
    printf("surface wind only (model will estimate winds aloft).\n\n");
    
    // Surface wind
    while (1) {
        printf("SURFACE WIND (0 FT): ");
        if (!fgets(input, sizeof(input), stdin)) exit(0);
        if (strcmp(trim_input(input), "QUIT") == 0) exit(0);
        
        float speed, direction;
        if (parse_wind_input(input, &speed, &direction)) {
            model->atmosphere.layers[0].speed_kts = speed;
            model->atmosphere.layers[0].direction_deg = direction;
            break;
        }
        printf("ERROR: Enter as SPEED@DIRECTION (e.g., 15@270).\n");
    }
    
    // Ask if user wants detailed wind profile
    printf("\nENTER DETAILED WIND PROFILE? (Y/N): ");
    if (fgets(input, sizeof(input), stdin)) {
        if (toupper(input[0]) == 'Y') {
            // Get winds at each altitude
            for (int i = 1; i < NUM_WIND_LAYERS; i++) {
                printf("WIND AT %.0f FT: ", WIND_ALTITUDES_FT[i]);
                if (fgets(input, sizeof(input), stdin)) {
                    float speed, direction;
                    if (parse_wind_input(input, &speed, &direction)) {
                        model->atmosphere.layers[i].speed_kts = speed;
                        model->atmosphere.layers[i].direction_deg = direction;
                    }
                }
            }
        } else {
            // Estimate winds aloft using realistic atmospheric model
            float surface_speed = model->atmosphere.layers[0].speed_kts;
            float surface_dir = model->atmosphere.layers[0].direction_deg;
            
            for (int i = 1; i < NUM_WIND_LAYERS; i++) {
                // Realistic wind profile based on atmospheric observations
                // Power law: V = V_surface * (h/h_ref)^alpha where alpha ≈ 0.14-0.20
                // But cap maximum speeds to realistic jet stream values (200-250 knots)
                float altitude_ft = WIND_ALTITUDES_FT[i];
                float speed_multiplier;
                
                if (altitude_ft <= 10000) {
                    // Lower atmosphere: gradual increase
                    speed_multiplier = 1.0f + (altitude_ft / 10000.0f) * 0.8f;  // Up to 1.8x at 10k ft
                } else if (altitude_ft <= 30000) {
                    // Mid-troposphere: approach jet stream level
                    speed_multiplier = 1.8f + ((altitude_ft - 10000.0f) / 20000.0f) * 1.2f;  // 1.8x to 3.0x
                } else {
                    // Upper troposphere: jet stream region, cap at realistic values
                    speed_multiplier = 3.0f + ((altitude_ft - 30000.0f) / 20000.0f) * 1.5f;  // 3.0x to 4.5x
                }
                
                float estimated_speed = surface_speed * speed_multiplier;
                
                // Cap at realistic maximum jet stream speeds (200-250 knots)
                // Allow slightly higher if surface wind is already strong
                float max_wind = 200.0f + (surface_speed * 0.5f);  // Dynamic cap
                if (max_wind > 250.0f) max_wind = 250.0f;
                
                model->atmosphere.layers[i].speed_kts = (estimated_speed > max_wind) ? max_wind : estimated_speed;
                
                // Direction veers with altitude (Ekman spiral, ~30° in upper levels)
                model->atmosphere.layers[i].direction_deg = surface_dir + (i * 5.0f);
                if (model->atmosphere.layers[i].direction_deg >= 360.0) {
                    model->atmosphere.layers[i].direction_deg -= 360.0;
                }
            }
            
            printf("  (Winds aloft estimated from surface conditions)\n");
        }
    }
    
    // Display wind profile
    printf("\nWIND PROFILE SUMMARY:\n");
    for (int i = 0; i < NUM_WIND_LAYERS; i++) {
        printf("  %6.0f ft: %5.1f kts @ %5.1f deg\n",
               WIND_ALTITUDES_FT[i],
               model->atmosphere.layers[i].speed_kts,
               model->atmosphere.layers[i].direction_deg);
    }
}

// Calculate cloud parameters using WSEG-10 formulas
void calculate_cloud_parameters(FalloutModel* model) {
    float yield_mt = model->weapon.yield_kt / 1000.0f;
    
    // Cloud top altitude: H_top = 12.5 * W^0.25 km
    model->cloud.cloud_top_km = 12.5f * powf(yield_mt, 0.25f);
    
    // Cloud base: approximately 1/3 of top altitude for surface burst
    if (model->weapon.is_surface_burst) {
        model->cloud.cloud_base_km = model->cloud.cloud_top_km * 0.3f;
    } else {
        model->cloud.cloud_base_km = model->weapon.hob_m / 1000.0f;
    }
    
    // Stem diameter: D = 1.5 * W^0.33 km
    model->cloud.stem_diameter_km = 1.5f * powf(yield_mt, 0.33f);
    
    // Stabilization time: t = 10 * W^0.25 minutes
    model->cloud.stabilization_time_min = 10.0f * powf(yield_mt, 0.25f);
    
    printf("\nCLOUD PARAMETERS:\n");
    printf("  Cloud top: %.1f km (%.0f ft)\n", 
           model->cloud.cloud_top_km, model->cloud.cloud_top_km * 3280.84);
    printf("  Cloud base: %.1f km (%.0f ft)\n",
           model->cloud.cloud_base_km, model->cloud.cloud_base_km * 3280.84);
    printf("  Stem diameter: %.1f km\n", model->cloud.stem_diameter_km);
    printf("  Stabilization time: %.1f minutes\n", model->cloud.stabilization_time_min);
}

// Calculate particle size distribution (log-normal WSEG-10)
void calculate_particle_distribution(FalloutModel* model) {
    // WSEG-10 typical values
    model->particles.median_microns = 120.0f;  // Median diameter
    model->particles.geometric_sigma = 2.5f;   // Log-normal spread
    
    if (!model->weapon.is_surface_burst) {
        // Air burst: much finer particles, minimal fallout
        model->particles.median_microns = 20.0f;
        model->particles.geometric_sigma = 3.0f;
    }
    
    // Calculate mass fractions for each size class
    float total_mass = 0.0f;
    for (int i = 0; i < NUM_PARTICLE_CLASSES; i++) {
        float d = PARTICLE_DIAMETERS[i];
        float ln_d = logf(d);
        float ln_median = logf(model->particles.median_microns);
        float ln_sigma = logf(model->particles.geometric_sigma);
        
        // Log-normal distribution: f(d) = (1/(d*σ*√(2π))) * exp(-0.5*((ln(d)-ln(d₀))/ln(σ))²)
        // Multiply by d³ for mass distribution
        float exponent = -0.5f * powf((ln_d - ln_median) / ln_sigma, 2.0f);
        model->particles.mass_fractions[i] = d * d * d * expf(exponent);
        total_mass += model->particles.mass_fractions[i];
    }
    
    // Normalize to sum to 1.0
    for (int i = 0; i < NUM_PARTICLE_CLASSES; i++) {
        model->particles.mass_fractions[i] /= total_mass;
    }
    
    printf("\nPARTICLE SIZE DISTRIBUTION:\n");
    printf("  Median diameter: %.0f microns\n", model->particles.median_microns);
    printf("  Geometric sigma: %.2f\n", model->particles.geometric_sigma);
    printf("\n  Size class distribution:\n");
    for (int i = 0; i < NUM_PARTICLE_CLASSES; i += 2) {
        printf("    %4.0f um: %5.1f%%", 
               PARTICLE_DIAMETERS[i], 
               model->particles.mass_fractions[i] * 100.0);
        if (i + 1 < NUM_PARTICLE_CLASSES) {
            printf("    %4.0f um: %5.1f%%\n",
                   PARTICLE_DIAMETERS[i+1],
                   model->particles.mass_fractions[i+1] * 100.0);
        } else {
            printf("\n");
        }
    }
}

// Calculate fallout pattern for entire grid
void calculate_fallout_pattern(FalloutModel* model) {
    // Process each particle size class
    for (int pc = 0; pc < NUM_PARTICLE_CLASSES; pc++) {
        calculate_transport_and_deposition(model, pc);
    }
    
    // Apply surface/air burst scaling
    float burst_scale = model->weapon.is_surface_burst ? 1.0f : 0.005f;
    
    for (int y = 0; y < GRID_SIZE; y++) {
        for (int x = 0; x < GRID_SIZE; x++) {
            model->grid[y][x].dose_rate_rhr *= burst_scale;
        }
    }
}

// Get particle settling velocity (Stokes law with Newton correction)
float get_settling_velocity(float diameter_microns, float altitude_m) {
    float d_m = diameter_microns * 1.0e-6f;  // Convert to meters

    // Air density decreases with altitude: ρ = ρ₀ * exp(-h/H), H ≈ 8500m
    float altitude_scale = 8500.0f;
    float rho_air = AIR_DENSITY * expf(-altitude_m / altitude_scale);

    // Stokes settling velocity: v = g*d²*(ρ_p - ρ_a)/(18*μ)
    float v_stokes = GRAVITY * d_m * d_m * (PARTICLE_DENSITY - rho_air) / (18.0f * AIR_VISCOSITY);

    // For larger particles, apply Newton drag correction
    // Reynolds number: Re = ρ*v*d/μ
    float re = rho_air * v_stokes * d_m / AIR_VISCOSITY;

    if (re < 1.0f) {
        // Pure Stokes regime
        return v_stokes;
    } else if (re < 1000.0f) {
        // Transition regime: reduce velocity
        float correction = 1.0f / (1.0f + 0.15f * powf(re, 0.687f));
        return v_stokes * correction;
    } else {
        // Newton regime: v ~ sqrt(d)
        return sqrtf(4.0f * GRAVITY * d_m * (PARTICLE_DENSITY - rho_air) / (3.0f * rho_air));
    }
}

// Get fission product activity (WSEG-10)
float get_fission_activity(float yield_kt, float time_hr, float fission_fraction) {
    // Reference: 1 MT fission at H+1 gives 1.6×10⁶ R/hr at 1 NM
    float fission_kt = yield_kt * fission_fraction;
    float fission_mt = fission_kt / 1000.0f;
    
    // Activity scales linearly with fission yield
    float activity_1hr = REF_ACTIVITY_1MT_1HR * fission_mt;
    
    // Way-Wigner decay: I(t) = I(1) * t^(-1.2)
    float activity = activity_1hr * powf(time_hr, -DECAY_EXPONENT);
    
    return activity;
}

// Calculate transport and deposition for one particle class
void calculate_transport_and_deposition(FalloutModel* model, int particle_class) {
    float diameter = PARTICLE_DIAMETERS[particle_class];
    float mass_fraction = model->particles.mass_fractions[particle_class];
    
    if (mass_fraction < 0.001f) return;  // Skip negligible classes
    
    // Get activity for this particle class
    float total_activity = get_fission_activity(
        model->weapon.yield_kt,
        model->time_since_burst_hr,
        model->weapon.fission_fraction
    );
    float class_activity = total_activity * mass_fraction;
    
    // Simulate particles released throughout cloud
    int num_release_points = 12; // fewer release points to reduce CPU / memory
    
    for (int release = 0; release < num_release_points; release++) {
        // Release altitude (distributed from base to top)
        float release_fraction = (float)release / (float)(num_release_points - 1);
        float release_altitude_m = (model->cloud.cloud_base_km + 
            release_fraction * (model->cloud.cloud_top_km - model->cloud.cloud_base_km)) * 1000.0f;
        
        // Fall time for this particle from this altitude
        float altitude = release_altitude_m;
        float total_drift_x = 0.0f;
        float total_drift_y = 0.0f;
        float total_fall_time_hr = 0.0f;
        
        // Integrate fall through atmosphere
        while (altitude > 0) {
            // Find wind layer for current altitude
            float altitude_ft = altitude * 3.28084f;
            int layer = 0;
            for (int i = 1; i < NUM_WIND_LAYERS; i++) {
                if (altitude_ft >= WIND_ALTITUDES_FT[i]) {
                    layer = i;
                }
            }
            
            float wind_speed_ms = model->atmosphere.layers[layer].speed_kts * KNOTS_TO_MS;
            float wind_dir_rad = model->atmosphere.layers[layer].direction_deg * M_PI / 180.0f;
            
            // Settling velocity at this altitude
            float v_settle = get_settling_velocity(diameter, altitude);
            
            // Time step: fall 500m
            float dh = (altitude > 500.0f) ? 500.0f : altitude;
            float dt_sec = dh / v_settle;
            float dt_hr = dt_sec / 3600.0f;
            
            // Drift during this time step (wind FROM direction, so add 180°)
            float transport_dir = wind_dir_rad + M_PI;
            total_drift_x += wind_speed_ms * dt_sec * cosf(transport_dir);
            total_drift_y += wind_speed_ms * dt_sec * sinf(transport_dir);
            
            total_fall_time_hr += dt_hr;
            altitude -= dh;
        }
        
        // Landing position (meters from GZ)
        float landing_x_m = total_drift_x;
        float landing_y_m = total_drift_y;
        
        // Convert to grid coordinates
        float landing_x_km = landing_x_m / 1000.0f;
        float landing_y_km = landing_y_m / 1000.0f;
        
        // Diffusion width (grows with transport distance)
        float transport_distance_km = sqrtf(landing_x_km * landing_x_km + landing_y_km * landing_y_km);
        float sigma_km = model->cloud.stem_diameter_km * 0.5f + 0.1f * transport_distance_km;
        
        // Deposit activity on grid using Gaussian distribution
        for (int gy = 0; gy < GRID_SIZE; gy++) {
            for (int gx = 0; gx < GRID_SIZE; gx++) {
                // Grid cell position relative to GZ (km)
                float cell_x_km = (gx - model->gz_x) * GRID_RESOLUTION_KM;
                float cell_y_km = (gy - model->gz_y) * GRID_RESOLUTION_KM;

                // Distance from landing point
                float dx = cell_x_km - landing_x_km;
                float dy = cell_y_km - landing_y_km;

                // Gaussian deposition
                float deposition = gaussian_plume(dx, dy, sigma_km, sigma_km);

                // Terrain effects on deposition
                float terrain_mod = 1.0f;
                if (model->terrain.active) {
                    float elev = get_terrain_elev(model, cell_x_km, cell_y_km);
                    // Higher elevation = earlier arrival, modified deposition
                    // Get terrain gradient for wind blocking effect
                    float elev_n = get_terrain_elev(model, cell_x_km, cell_y_km + GRID_RESOLUTION_KM);
                    float elev_s = get_terrain_elev(model, cell_x_km, cell_y_km - GRID_RESOLUTION_KM);
                    float elev_e = get_terrain_elev(model, cell_x_km + GRID_RESOLUTION_KM, cell_y_km);
                    float elev_w = get_terrain_elev(model, cell_x_km - GRID_RESOLUTION_KM, cell_y_km);

                    // Wind direction (transport TO)
                    float wdir = (model->atmosphere.layers[0].direction_deg + 180.0f) * M_PI / 180.0f;
                    // Gradient in wind direction
                    float grad_x = (elev_e - elev_w) / (2.0f * GRID_RESOLUTION_KM * 1000.0f);
                    float grad_y = (elev_n - elev_s) / (2.0f * GRID_RESOLUTION_KM * 1000.0f);
                    float slope_factor = grad_x * cosf(wdir) + grad_y * sinf(wdir);

                    // Leeward slopes: enhanced deposition (1.5x), Windward: reduced (0.6x)
                    terrain_mod = 1.0f - slope_factor * 50.0f;
                    if (terrain_mod < 0.6f) terrain_mod = 0.6f;
                    if (terrain_mod > 1.5f) terrain_mod = 1.5f;

                    // Valley concentration effect (low spots collect more)
                    float local_avg = (elev_n + elev_s + elev_e + elev_w) / 4.0f;
                    if (elev < local_avg - 50.0f) terrain_mod *= 1.3f; // Valley
                    if (elev > local_avg + 100.0f) terrain_mod *= 0.8f; // Ridge
                }

                // Add activity (scaled by release point weight)
                float activity_contribution = class_activity * deposition / num_release_points * terrain_mod;

                // Convert from point source to area dose rate
                // Assume activity spreads over 1 km²
                float dose_rate = activity_contribution / (M_PI * sigma_km * sigma_km);

                model->grid[gy][gx].dose_rate_rhr += dose_rate;
                
                // Track arrival time (adjusted for terrain elevation)
                float terrain_elev_km = model->terrain.active ?
                    get_terrain_elev(model, cell_x_km, cell_y_km) / 1000.0f : 0.0f;
                float elev_time_adj = terrain_elev_km / (get_settling_velocity(diameter, 0) * 3.6f);
                float arrival_time = model->cloud.stabilization_time_min / 60.0f + total_fall_time_hr - elev_time_adj;
                if (arrival_time < 0.1f) arrival_time = 0.1f;
                if (model->grid[gy][gx].arrival_time_hr == 0.0f || arrival_time < model->grid[gy][gx].arrival_time_hr) {
                    model->grid[gy][gx].arrival_time_hr = arrival_time;
                }
                
                // Track dominant particle size
                if (activity_contribution > 0.0f) {
                    model->grid[gy][gx].particle_class_max = particle_class;
                }
            }
        }
    }
}

// Gaussian plume function
float gaussian_plume(float x, float y, float sigma_x, float sigma_y) {
    float term_x = x * x / (2.0f * sigma_x * sigma_x);
    float term_y = y * y / (2.0f * sigma_y * sigma_y);
    return expf(-(term_x + term_y));
}

// Display all results
void display_results(FalloutModel* model) {
    printf("\n\n");
    printf("========================================================\n");
    printf("                  FALLOUT PREDICTION RESULTS\n");
    printf("========================================================\n");
    
    display_dose_map(model);
    display_contours(model);
    display_sector_report(model);
    display_timeline(model);
}

// Display ASCII dose rate map
void display_dose_map(FalloutModel* model) {
    printf("\n\n*** DOSE RATE MAP (R/HR AT H+1) ***\n\n");
    printf("Legend:\n");
    printf("  . = <1 R/hr      : = 1-10 R/hr    + = 10-50 R/hr\n");
    printf("  * = 50-100 R/hr  # = 100-500 R/hr @ = 500-1000 R/hr\n");
    printf("  X = >1000 R/hr   G = Ground Zero\n");
    printf("\nGrid: %dx%d cells, %.1f km/cell (%.0f x %.0f km total)\n",
           GRID_SIZE, GRID_SIZE, GRID_RESOLUTION_KM,
           GRID_SIZE * GRID_RESOLUTION_KM, GRID_SIZE * GRID_RESOLUTION_KM);
    printf("\n");
    
    // Display every 2nd cell for 40-column output
    int step = 2;
    
    // Column headers
    printf("     ");
    for (int x = 0; x < GRID_SIZE; x += step) {
        printf("%d", (x / 10) % 10);
    }
    printf("\n");
    
    for (int y = 0; y < GRID_SIZE; y += step) {
        printf("%3d: ", y);
        
        for (int x = 0; x < GRID_SIZE; x += step) {
            if (x == model->gz_x && y == model->gz_y) {
                printf("G");
                continue;
            }
            
            float dose = model->grid[y][x].dose_rate_rhr;
            char symbol;
            
            if (dose < 1.0f) symbol = '.';
            else if (dose < 10.0f) symbol = ':';
            else if (dose < 50.0f) symbol = '+';
            else if (dose < 100.0f) symbol = '*';
            else if (dose < 500.0f) symbol = '#';
            else if (dose < 1000.0f) symbol = '@';
            else symbol = 'X';
            
            printf("%c", symbol);
        }
        printf("\n");
    }
}

// Display dose rate contours
void display_contours(FalloutModel* model) {
    printf("\n\n*** DOSE RATE CONTOURS ***\n\n");
    
    float contour_levels[] = {10.0f, 50.0f, 100.0f, 500.0f, 1000.0f, 5000.0f};
    int num_contours = sizeof(contour_levels) / sizeof(contour_levels[0]);
    
    printf("Contour level | Max extent (km) | Area (sq km)\n");
    printf("----------------------------------------------\n");
    
    for (int c = 0; c < num_contours; c++) {
        float level = contour_levels[c];
        float max_distance = 0.0f;
        int cell_count = 0;
        
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                if (model->grid[y][x].dose_rate_rhr >= level) {
                    cell_count++;
                    
                    float dx = (x - model->gz_x) * GRID_RESOLUTION_KM;
                    float dy = (y - model->gz_y) * GRID_RESOLUTION_KM;
                    float dist = sqrtf(dx * dx + dy * dy);
                    
                    if (dist > max_distance) {
                        max_distance = dist;
                    }
                }
            }
        }
        
        float area_km2 = cell_count * GRID_RESOLUTION_KM * GRID_RESOLUTION_KM;
        
        printf("%8.0f R/hr |    %6.1f km   | %8.1f sq km\n",
               level, max_distance, area_km2);
    }
}

// Display sector-by-sector report
void display_sector_report(FalloutModel* model) {
    printf("\n\n*** DOWNWIND SECTOR ANALYSIS ***\n\n");
    
    // Find primary wind direction (surface)
    float wind_dir = model->atmosphere.layers[0].direction_deg;
    float downwind_bearing = fmodf(wind_dir + 180.0f, 360.0f);
    
    printf("Primary transport direction: %.0f degrees (TO)\n", downwind_bearing);
    printf("\nDistance | Max dose | Avg dose | Arrival time\n");
    printf("(km)     | (R/hr)   | (R/hr)   | (hours)\n");
    printf("------------------------------------------------\n");
    
    for (int range_km = 5; range_km <= 150; range_km += 5) {
        float max_dose = 0.0f;
        float avg_dose = 0.0f;
        float min_arrival = 999.0f;
        int count = 0;
        
        // Sample cells in this range band (±2 km)
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                float dx = (x - model->gz_x) * GRID_RESOLUTION_KM;
                float dy = (y - model->gz_y) * GRID_RESOLUTION_KM;
                float dist = sqrtf(dx * dx + dy * dy);
                
                if (dist >= range_km - 2 && dist <= range_km + 2) {
                    float dose = model->grid[y][x].dose_rate_rhr;
                    if (dose > max_dose) max_dose = dose;
                    avg_dose += dose;
                    count++;
                    
                    if (model->grid[y][x].arrival_time_hr > 0 &&
                        model->grid[y][x].arrival_time_hr < min_arrival) {
                        min_arrival = model->grid[y][x].arrival_time_hr;
                    }
                }
            }
        }
        
        if (count > 0) {
            avg_dose /= count;
        }
        
        if (max_dose >= 1.0) {  // Only show significant ranges
            printf("%5d    | %8.1f | %8.1f | H+%.1f hr\n",
                   range_km, max_dose, avg_dose,
                   (min_arrival < 900) ? min_arrival : 0.0);
        }
    }
}

// Display timeline of fallout arrival
void display_timeline(FalloutModel* model) {
    printf("\n\n*** FALLOUT ARRIVAL TIMELINE ***\n\n");
    printf("Time bins showing area coverage by arrival time:\n\n");
    
    printf("Time range    | Area affected (sq km) | Peak dose (R/hr)\n");
    printf("----------------------------------------------------------\n");
    
    float time_bins[] = {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 12.0f, 24.0f};
    int num_bins = sizeof(time_bins) / sizeof(time_bins[0]);
    
    for (int b = 0; b < num_bins; b++) {
        float t_start = (b == 0) ? 0.0f : time_bins[b-1];
        float t_end = time_bins[b];

        int cell_count = 0;
        float max_dose = 0.0f;
        
        for (int y = 0; y < GRID_SIZE; y++) {
            for (int x = 0; x < GRID_SIZE; x++) {
                float arrival = model->grid[y][x].arrival_time_hr;
                float dose = model->grid[y][x].dose_rate_rhr;
                
                if (arrival >= t_start && arrival < t_end && dose >= 1.0) {
                    cell_count++;
                    if (dose > max_dose) max_dose = dose;
                }
            }
        }
        
        float area = cell_count * GRID_RESOLUTION_KM * GRID_RESOLUTION_KM;
        
        printf("H+%.1f-%.1f hr |      %8.1f sq km   |   %8.1f\n",
               t_start, t_end, area, max_dose);
    }
}

// Utility: trim whitespace from input
char* trim_input(char* str) {
    while (isspace(*str)) str++;
    char* end = str + strlen(str) - 1;
    while (end > str && isspace(*end)) {
        *end = '\0';
        end--;
    }
    
    // Convert to uppercase
    for (char* p = str; *p; p++) {
        *p = toupper(*p);
    }
    
    return str;
}

// Parse wind input (SPEED@DIRECTION)
int parse_wind_input(const char* input, float* speed, float* direction) {
    char buffer[80];
    strncpy(buffer, input, sizeof(buffer) - 1);
    buffer[sizeof(buffer) - 1] = '\0';
    
    char* at_sign = strchr(buffer, '@');
    if (at_sign) {
        *at_sign = '\0';
        
        if (sscanf(buffer, "%f", speed) == 1 &&
            sscanf(at_sign + 1, "%f", direction) == 1) {
            return 1;
        }
    }
    
    return 0;
}
