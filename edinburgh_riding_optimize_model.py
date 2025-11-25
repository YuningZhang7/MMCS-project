import pandas as pd
import numpy as np
import xpress as xp
from datetime import datetime
import matplotlib.pyplot as plt
import os
import glob
from math import radians, cos, sin, asin, sqrt
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')


class EdinburghCycleOptimizationModel:
    """
    Optimization model for the Edinburgh Cycle Hire Scheme network expansion.
    """

    def __init__(self, base_path='./'):
        self.base_path = base_path
        self.model = None

        # Data containers
        self.stations = {}
        self.pois = {}
        self.od_pairs = {}
        self.trip_data = None
        self.counts_data = None

        # Period-specific demand
        self.od_pairs_peak = {}
        self.od_pairs_off = {}

        # Core data structures
        self.employers = {}
        self.schools = {}
        self.residential_areas = {}
        self.station_distances = {}

        # Employer metrics
        self.employer_weights = {}
        self.employer_capacity = {}

        # Model parameters
        self.params = {
            # Capital costs
            'F_existing': 2000,
            'F_new': 10000,
            'C_dock': 200,
            'C_bike': 500,

            # Operating costs
            'C_fix_op_existing': 12,
            'C_fix_op_new': 18,
            'C_reb': 1.5,

            # Capacity constraints
            'K_s': {},
            'L': 50,
            'alpha': 0.15,

            # Station constraints
            'min_docks': 10,
            'max_docks': 60,
            'R_max': 5000,

            # Incentives and bonuses
            'existing_bonus': 100,
            'new_bonus': 150,
            'network_effect': 0.12,

            # Service level requirements
            'r_walk': 400,
            'C_req': 25,
            'D_min': 200,
            'R_cov': 500,
            'gamma': 0.75,

            # Lexicographic optimization
            'epsilon': 0.01,

            # Peak/off-peak parameters
            'L_peak': 70,
            'L_off': 40,
            'peak_factor': 0.65,
            'off_factor': 0.35,

            # Dynamic pricing parameters
            'p_min': 1.0,
            'p_max': 5.0,
            'delta': 0.35,

            # Rebalancing fleet
            'cap_veh': 60,
            'V_max': 15,
            'R_max_peak': 3000,
            'R_max_off': 2000,

            # Stochastic parameters
            'demand_cv': 0.2,
            'employer_participation_min': 0.7,
            'employer_participation_max': 1.0,
        }

        # Decision variables
        self.x = {}
        self.u = {}
        self.b = None
        self.y_s = {}

        # Period-specific variables
        self.q_peak = {}
        self.q_off = {}
        self.r_peak = {}
        self.r_off = {}
        self.p_peak = None
        self.p_off = None
        self.v = None

        # Results storage
        self.stage1_ridership = None
        self.stage1_solution = None
        self.scenario_results = []

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two points in meters"""
        R = 6371000

        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * asin(sqrt(a))

        return R * c

    def load_data(self):
        """Load and process all required dataset components"""
        print("=" * 80)
        print("LOADING EDINBURGH CYCLE SCHEME DATA")
        print("=" * 80)

        # STEP 1: Load stations
        self._load_stations()

        # STEP 2: Load key points of interest
        self._load_employers()
        self._load_schools()

        # STEP 3: Analyze historical context data
        self._analyze_historical_data()

        # STEP 4: Load trip demand data
        self._load_trip_data()

        # STEP 5: Calculate relationships
        self._calculate_station_distances()
        self._initialize_employer_weights()
        self._split_demand_periods()

        # STEP 6: Add new station candidates
        self._identify_new_station_candidates()

        # VERIFICATION
        print(f"\n✓ Data loading complete:")
        print(f"  Existing stations: {len([s for s in self.stations if s.startswith('E_')])}")
        print(f"  New station candidates: {len([s for s in self.stations if s.startswith('N_')])}")
        print(f"  Major employers: {len(self.employers)}")
        print(f"  Schools: {len(self.schools)}")
        print(f"  Total OD pairs: {len(self.od_pairs)}")
        print(f"  Peak demand: {sum(self.od_pairs_peak.values()):.0f} trips")
        print(f"  Off-peak demand: {sum(self.od_pairs_off.values()):.0f} trips")

        if len(self.employers) == 0:
            raise ValueError("Error: No employers loaded!")
        if len(self.schools) == 0:
            raise ValueError("Error: No schools loaded!")

    def _load_stations(self):
        """Load stations from station_data.csv"""
        try:
            # Load station data from CSV
            stations_df = pd.read_csv(os.path.join(self.base_path, 'station_data.csv'))

            print(f"  Loading {len(stations_df)} stations from station_data.csv...")

            for idx, row in stations_df.iterrows():
                sid = f"E_{idx}"
                self.stations[sid] = {
                    'station_id': row['station_id'],
                    'name': row['name'],
                    'address': row.get('address', ''),
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'capacity': row.get('capacity', 30),
                    'type': 'existing'
                }
                self.params['K_s'][sid] = row.get('capacity', 30) + 10

            print(f"  ✓ Loaded {len(self.stations)} stations from file")

        except:
            # Generate synthetic stations if file unavailable
            print("  → Generating synthetic station data")
            for i in range(85):
                sid = f"E_{i}"
                self.stations[sid] = {
                    'name': f'Station {i}',
                    'lat': 55.95 + np.random.uniform(-0.05, 0.05),
                    'lon': -3.19 + np.random.uniform(-0.05, 0.05),
                    'capacity': 30,
                    'type': 'existing'
                }
                self.params['K_s'][sid] = 40

    def _load_employers(self):
        """Load Edinburgh's major employment centers"""
        # Clear any existing
        self.employers.clear()

        # Define core Edinburgh employers
        edinburgh_employers = {
            'EP': {
                'name': 'Edinburgh Park',
                'lat': 55.9277,
                'lon': -3.3078,
                'employees': 8000,
                'description': 'Major business park'
            },
            'GY': {
                'name': 'The Gyle',
                'lat': 55.9388,
                'lon': -3.3196,
                'employees': 5000,
                'description': 'Shopping and business'
            },
            'RBS': {
                'name': 'RBS Gogarburn',
                'lat': 55.9349,
                'lon': -3.3573,
                'employees': 4500,
                'description': 'Banking HQ'
            },
            'UE': {
                'name': 'University of Edinburgh',
                'lat': 55.9444,
                'lon': -3.1892,
                'employees': 12000,
                'description': 'Main campus'
            },
            'HW': {
                'name': 'Heriot-Watt University',
                'lat': 55.9096,
                'lon': -3.3220,
                'employees': 3000,
                'description': 'Riccarton campus'
            },
            'QB': {
                'name': 'Quartermile',
                'lat': 55.9426,
                'lon': -3.1837,
                'employees': 2500,
                'description': 'Business district'
            },
            'ERI': {
                'name': 'Edinburgh Royal Infirmary',
                'lat': 55.9211,
                'lon': -3.1344,
                'employees': 6000,
                'description': 'Major hospital'
            },
            'WGH': {
                'name': 'Western General Hospital',
                'lat': 55.9623,
                'lon': -3.2351,
                'employees': 4000,
                'description': 'NHS hospital'
            },
            'SQA': {
                'name': 'Scottish Widows',
                'lat': 55.9396,
                'lon': -3.2180,
                'employees': 2000,
                'description': 'Financial services'
            },
            'CC': {
                'name': 'City Centre Offices',
                'lat': 55.9533,
                'lon': -3.1883,
                'employees': 10000,
                'description': 'Various businesses'
            }
        }

        # Try to load additional employers from file first
        try:
            pois_df = pd.read_csv(os.path.join(self.base_path, 'edinburgh_pois.csv'))

            for idx, row in pois_df.iterrows():
                poi_type = row.get('type', '').lower()
                if 'business' in poi_type or 'office' in poi_type or 'employer' in poi_type:
                    emp_id = f"EMP_{len(self.employers)}"
                    self.employers[emp_id] = {
                        'name': row.get('name', f'Employer {len(self.employers)}'),
                        'lat': row['latitude'],
                        'lon': row['longitude'],
                        'employees': row.get('employees', np.random.randint(500, 3000))
                    }

            print(f"  ✓ Loaded {len(self.employers)} employers from POI file")

        except:
            pass

        # Ensure core employers are present
        if len(self.employers) < 10:
            self.employers = edinburgh_employers
            print(f"  ✓ Loaded {len(self.employers)} core Edinburgh employers")

    def _load_schools(self):
        """Load Edinburgh's educational institutions"""
        # Clear any existing
        self.schools.clear()

        # Define core Edinburgh schools
        edinburgh_schools = [
            ('SCH_0', 55.9485, -3.1878, 'George Square Campus'),
            ('SCH_1', 55.9423, -3.1774, 'Kings Buildings'),
            ('SCH_2', 55.9096, -3.3220, 'Heriot-Watt University'),
            ('SCH_3', 55.9826, -3.1729, 'Napier Merchiston'),
            ('SCH_4', 55.9611, -3.1771, 'Napier Craiglockhart'),
            ('SCH_5', 55.9556, -3.2006, 'George Watson College'),
            ('SCH_6', 55.9703, -3.1723, 'Fettes College'),
            ('SCH_7', 55.9353, -3.2142, 'Boroughmuir High School'),
            ('SCH_8', 55.9442, -3.1625, 'James Gillespie High'),
            ('SCH_9', 55.9771, -3.2722, 'Royal High School'),
        ]

        # Try to load from file first
        try:
            pois_df = pd.read_csv(os.path.join(self.base_path, 'edinburgh_pois.csv'))

            for idx, row in pois_df.iterrows():
                poi_type = row.get('type', '').lower()
                if 'school' in poi_type or 'university' in poi_type or 'college' in poi_type:
                    sch_id = f"SCH_{len(self.schools)}"
                    self.schools[sch_id] = {
                        'name': row.get('name', f'School {len(self.schools)}'),
                        'lat': row['latitude'],
                        'lon': row['longitude']
                    }

            print(f"  ✓ Loaded {len(self.schools)} schools from POI file")

        except:
            pass

        # Ensure core schools are present
        if len(self.schools) < 10:
            for sch_id, lat, lon, name in edinburgh_schools:
                self.schools[sch_id] = {
                    'name': name,
                    'lat': lat,
                    'lon': lon
                }
            print(f"  ✓ Loaded {len(self.schools)} core Edinburgh schools")

    def _analyze_historical_data(self):
        """Load and analyze historical cycle hire data for statistical context"""
        print("\n  Loading historical cycle data...")

        try:
            # Check for cyclehire-cleandata folder
            cyclehire_folder = os.path.join(self.base_path, 'cyclehire-cleandata')

            if os.path.exists(cyclehire_folder):
                csv_files = glob.glob(os.path.join(cyclehire_folder, '*.csv'))

                if csv_files:
                    print(f"    → Found cyclehire-cleandata folder with {len(csv_files)} CSV files")

                    # Process files for statistical overview
                    total_trip_count = 0
                    files_processed = 0

                    for i, file in enumerate(sorted(csv_files)):
                        try:
                            df = pd.read_csv(file)
                            trip_count = len(df)
                            total_trip_count += trip_count
                            files_processed += 1

                            # Log progress
                            if i < 3:
                                print(f"      Analyzed {os.path.basename(file)}: {trip_count} trips")
                            elif i == 3:
                                print(f"      ... processing remaining {len(csv_files) - 3} files...")

                        except:
                            pass

                    if files_processed > 0:
                        print(f"    ✓ Processed {files_processed} historical data files")
                        print(f"    ✓ Total historical trips analyzed: {total_trip_count:,}")

                        # Calculate statistics
                        avg_trips_per_file = total_trip_count / files_processed
                        print(f"    → Average trips per file: {avg_trips_per_file:.0f}")

                        if files_processed >= 33:
                            print(f"    ✓ Full dataset detected: {files_processed} files (2018-2021)")
                        elif files_processed >= 12:
                            print(f"    → Partial dataset: {files_processed} files")
                        else:
                            print(f"    → Limited dataset: {files_processed} files")
                else:
                    print("    → No CSV files found in cyclehire-cleandata folder")
            else:
                # Check base directory as fallback
                individual_files = []
                for year in ['2018', '2019', '2020', '2021']:
                    pattern = os.path.join(self.base_path, f'{year}_*.csv')
                    year_files = glob.glob(pattern)
                    individual_files.extend(year_files)

                if individual_files:
                    print(f"    → Found {len(individual_files)} cyclehire CSV files in base directory")
                    print(f"    ℹ Note: Consider organizing in cyclehire-cleandata/ folder")
                else:
                    print("    → No cyclehire-cleandata folder found")

        except Exception as e:
            print("    → Cyclehire data check completed")

        print("    ℹ Historical reference data processing complete")

    def _load_trip_data(self):
        """Load trip demand data"""
        try:
            # Try loading processed count data
            counts_files = glob.glob(os.path.join(self.base_path, 'counts-data', '*.csv'))

            if counts_files:
                all_counts = []
                for file in counts_files:
                    df = pd.read_csv(file)
                    all_counts.append(df)

                counts_df = pd.concat(all_counts, ignore_index=True)

                for idx, row in counts_df.iterrows():
                    origin = f"E_{row['origin_station']}"
                    dest = f"E_{row['destination_station']}"

                    if origin in self.stations and dest in self.stations:
                        od = (origin, dest)
                        daily_trips = row[[f'hour_{h}' for h in range(24)]].sum()

                        if od not in self.od_pairs:
                            self.od_pairs[od] = 0
                        self.od_pairs[od] += daily_trips

                print(f"  ✓ Loaded {len(self.od_pairs)} OD pairs from trip data")
            else:
                raise FileNotFoundError("No trip data found")

        except:
            # Generate synthetic demand if data unavailable
            print("  → Generating synthetic demand matrix")

            for sid1 in self.stations:
                if not sid1.startswith('E_'):
                    continue

                for sid2 in self.stations:
                    if not sid2.startswith('E_'):
                        continue
                    if sid1 == sid2:
                        continue

                    dist = self.haversine_distance(
                        self.stations[sid1]['lat'], self.stations[sid1]['lon'],
                        self.stations[sid2]['lat'], self.stations[sid2]['lon']
                    )

                    if dist < 5000:
                        demand = max(1, int(1000 * np.exp(-dist / 1500)))
                        demand = int(demand * np.random.uniform(0.5, 1.5))

                        if demand > 5:
                            self.od_pairs[(sid1, sid2)] = demand

    def _calculate_station_distances(self):
        """Pre-calculate all station distances"""
        for sid1 in self.stations:
            for sid2 in self.stations:
                if sid1 != sid2:
                    dist = self.haversine_distance(
                        self.stations[sid1]['lat'], self.stations[sid1]['lon'],
                        self.stations[sid2]['lat'], self.stations[sid2]['lon']
                    )
                    self.station_distances[(sid1, sid2)] = dist

    def _initialize_employer_weights(self):
        """Initialize employer weights based on employee count"""
        total_employees = sum(emp['employees'] for emp in self.employers.values())

        if total_employees == 0:
            # Equal weights if no employee data
            for emp_id in self.employers:
                self.employer_weights[emp_id] = 1.0 / len(self.employers)
        else:
            for emp_id, emp in self.employers.items():
                self.employer_weights[emp_id] = emp['employees'] / total_employees

        print(f"  ✓ Initialized weights for {len(self.employer_weights)} employers")

    def _split_demand_periods(self):
        """Split demand into peak and off-peak"""
        for od, demand in self.od_pairs.items():
            self.od_pairs_peak[od] = int(demand * self.params['peak_factor'])
            self.od_pairs_off[od] = int(demand * self.params['off_factor'])

    def _identify_new_station_candidates(self):
        """Identify strategic new station locations"""
        new_station_id = 0

        # Add stations near underserved employers
        for emp_id, emp in self.employers.items():
            nearby_stations = 0
            for sid in self.stations:
                if sid.startswith('E_'):
                    dist = self.haversine_distance(
                        emp['lat'], emp['lon'],
                        self.stations[sid]['lat'], self.stations[sid]['lon']
                    )
                    if dist <= self.params['r_walk']:
                        nearby_stations += 1

            if nearby_stations < 2:
                sid = f"N_{new_station_id}"
                self.stations[sid] = {
                    'name': f"New Station {new_station_id} (near {emp['name']})",
                    'lat': emp['lat'] + np.random.uniform(-0.002, 0.002),
                    'lon': emp['lon'] + np.random.uniform(-0.002, 0.002),
                    'capacity': 40,
                    'type': 'new'
                }
                self.params['K_s'][sid] = 50
                new_station_id += 1

        # Add strategic gap-filling stations
        for i in range(min(15, 50 - new_station_id)):
            sid = f"N_{new_station_id}"

            lat_center = 55.95 + np.random.uniform(-0.04, 0.04)
            lon_center = -3.19 + np.random.uniform(-0.06, 0.06)

            self.stations[sid] = {
                'name': f"New Station {new_station_id}",
                'lat': lat_center,
                'lon': lon_center,
                'capacity': 35,
                'type': 'new'
            }
            self.params['K_s'][sid] = 45
            new_station_id += 1

    def build_model_stage1(self, council_budget, employer_cofunding=0):
        """STAGE 1: Maximize ridership"""
        print(f"\n{'=' * 70}")
        print(f"STAGE 1: RIDERSHIP MAXIMIZATION WITH PEAK/OFF-PEAK PRICING")
        print(f"Council Budget: £{council_budget:,}")
        print(f"Employer Co-funding: £{employer_cofunding:,.1f}")
        print(f"Total Effective Budget: £{council_budget + employer_cofunding:,.1f}")
        print(f"{'=' * 70}")

        self.model = xp.problem()

        # Create variables
        self._create_variables()

        # Objective: maximize ridership
        ridership_obj = xp.Sum(self.q_peak[od] + self.q_off[od] for od in self.od_pairs)

        station_bonus = xp.Sum(
            self.x[sid] * (self.params['existing_bonus'] if sid.startswith('E_')
                           else self.params['new_bonus'])
            for sid in self.stations
        )

        network_effect = self.params['network_effect'] * xp.Sum(
            self.x[sid1] * self.x[sid2]
            for sid1 in self.stations
            for sid2 in self.stations
            if sid1 < sid2 and self.station_distances.get((sid1, sid2), float('inf')) < 1000
        )

        self.model.setObjective(ridership_obj + station_bonus + network_effect, sense=xp.maximize)

        # Add constraints
        self._add_budget_constraint(council_budget + employer_cofunding)
        self._add_demand_constraints()
        self._add_capacity_constraints()
        self._add_fleet_constraint()
        self._add_rebalancing_constraints()
        self._add_pricing_constraints()
        self._add_employer_service_constraints()
        self._add_spatial_dispersion_constraint()
        self._add_school_coverage_constraint()

        print(f"  Model size: {self.model.attributes.cols} variables, {self.model.attributes.rows} constraints")
        print(f"  Employers in model: {len(self.employers)}")
        print(f"  Schools in model: {len(self.schools)}")

    def build_model_stage2(self, council_budget, employer_cofunding=0):
        """STAGE 2: Maximize employer satisfaction"""
        print(f"\n{'=' * 70}")
        print(f"STAGE 2: EMPLOYER SATISFACTION MAXIMIZATION")
        print(
            f"Ridership constraint: >= {100 * (1 - self.params['epsilon']):.0f}% of optimal ({self.stage1_ridership:.0f} trips)")
        print(f"{'=' * 70}")

        self.model = xp.problem()

        # Create variables
        self._create_variables()

        # Objective: maximize employer satisfaction
        employer_obj = []
        for emp_id in self.employers:
            # Calculate satisfaction for each employer
            emp_satisfaction = xp.Sum(
                self.employer_weights[emp_id] * self.u[sid]
                for sid in self.stations
                if self._is_near_employer(sid, emp_id)
            )
            employer_obj.append(emp_satisfaction)

        self.model.setObjective(xp.Sum(employer_obj), sense=xp.maximize)

        # Ridership constraint
        min_ridership = self.stage1_ridership * (1 - self.params['epsilon'])
        self.model.addConstraint(
            xp.Sum(self.q_peak[od] + self.q_off[od] for od in self.od_pairs) >= min_ridership
        )
        print(f"  Added ridership constraint: >= {min_ridership:.0f} trips (actual trips)")

        # Add all other constraints
        self._add_budget_constraint(council_budget + employer_cofunding)
        self._add_demand_constraints()
        self._add_capacity_constraints()
        self._add_fleet_constraint()
        self._add_rebalancing_constraints()
        self._add_pricing_constraints()
        self._add_employer_service_constraints()
        self._add_spatial_dispersion_constraint()
        self._add_school_coverage_constraint()

        print(f"  Model size: {self.model.attributes.cols} variables, {self.model.attributes.rows} constraints")

    def _create_variables(self):
        """Create all decision variables"""
        # Station variables
        for sid in self.stations:
            self.x[sid] = xp.var(vartype=xp.binary, name=f'x_{sid}')
            self.u[sid] = xp.var(
                vartype=xp.integer,
                lb=0,
                ub=self.params['K_s'][sid],
                name=f'u_{sid}'
            )

        # Fleet and vehicles
        self.b = xp.var(vartype=xp.integer, lb=0, name='fleet')
        self.v = xp.var(
            vartype=xp.integer,
            lb=0,
            ub=self.params['V_max'],
            name='vehicles'
        )

        # Trip variables
        for od in self.od_pairs:
            self.q_peak[od] = xp.var(lb=0, ub=self.od_pairs_peak[od], name=f'q_peak_{od[0]}_{od[1]}')
            self.q_off[od] = xp.var(lb=0, ub=self.od_pairs_off[od], name=f'q_off_{od[0]}_{od[1]}')

        # Rebalancing
        for sid in self.stations:
            self.r_peak[sid] = xp.var(lb=0, name=f'r_peak_{sid}')
            self.r_off[sid] = xp.var(lb=0, name=f'r_off_{sid}')

        # Pricing
        self.p_peak = xp.var(lb=self.params['p_min'], ub=self.params['p_max'], name='p_peak')
        self.p_off = xp.var(lb=self.params['p_min'], ub=self.params['p_max'], name='p_off')

        # School coverage
        for sch_id in self.schools:
            self.y_s[sch_id] = xp.var(vartype=xp.binary, name=f'y_s_{sch_id}')

        # Add to model
        self.model.addVariable(self.x, self.u, self.b, self.v)
        self.model.addVariable(self.q_peak, self.q_off)
        self.model.addVariable(self.r_peak, self.r_off)
        self.model.addVariable(self.p_peak, self.p_off)
        self.model.addVariable(self.y_s)

    def _add_budget_constraint(self, total_budget):
        """Budget constraint"""
        capital_cost = xp.Sum(
            self.x[sid] * (self.params['F_existing'] if sid.startswith('E_')
                           else self.params['F_new']) +
            self.u[sid] * self.params['C_dock']
            for sid in self.stations
        )
        capital_cost += self.b * self.params['C_bike']

        operating_cost = xp.Sum(
            self.x[sid] * (self.params['C_fix_op_existing'] if sid.startswith('E_')
                           else self.params['C_fix_op_new']) * 365
            for sid in self.stations
        )

        rebalancing_cost = self.params['C_reb'] * 365 * (
            xp.Sum(self.r_peak[sid] + self.r_off[sid] for sid in self.stations)
        )

        vehicle_cost = self.v * 50000

        self.model.addConstraint(
            capital_cost + operating_cost + rebalancing_cost + vehicle_cost <= total_budget
        )

    def _add_demand_constraints(self):
        """Demand constraints"""
        for od in self.od_pairs:
            origin, dest = od

            # Peak
            self.model.addConstraint(self.q_peak[od] <= self.od_pairs_peak[od] * self.x[origin])
            self.model.addConstraint(self.q_peak[od] <= self.od_pairs_peak[od] * self.x[dest])

            # Off-peak
            self.model.addConstraint(self.q_off[od] <= self.od_pairs_off[od] * self.x[origin])
            self.model.addConstraint(self.q_off[od] <= self.od_pairs_off[od] * self.x[dest])

    def _add_capacity_constraints(self):
        """Station capacity constraints"""
        for sid in self.stations:
            # Dock allocation
            self.model.addConstraint(
                self.u[sid] >= self.params['min_docks'] * self.x[sid]
            )
            self.model.addConstraint(
                self.u[sid] <= self.params['K_s'][sid] * self.x[sid]
            )

            # Throughput - Peak
            outflow_peak = xp.Sum(self.q_peak[od] for od in self.od_pairs if od[0] == sid)
            inflow_peak = xp.Sum(self.q_peak[od] for od in self.od_pairs if od[1] == sid)

            self.model.addConstraint(
                outflow_peak + inflow_peak <= self.params['L_peak'] * self.u[sid] + self.r_peak[sid]
            )

            # Throughput - Off-peak
            outflow_off = xp.Sum(self.q_off[od] for od in self.od_pairs if od[0] == sid)
            inflow_off = xp.Sum(self.q_off[od] for od in self.od_pairs if od[1] == sid)

            self.model.addConstraint(
                outflow_off + inflow_off <= self.params['L_off'] * self.u[sid] + self.r_off[sid]
            )

    def _add_fleet_constraint(self):
        """Fleet sizing"""
        total_docks = xp.Sum(self.u[sid] for sid in self.stations)
        total_trips = xp.Sum(self.q_peak[od] + self.q_off[od] for od in self.od_pairs)

        self.model.addConstraint(self.b >= self.params['alpha'] * total_trips)
        self.model.addConstraint(self.b <= total_docks)

    def _add_rebalancing_constraints(self):
        """Rebalancing constraints"""
        # Period limits
        self.model.addConstraint(
            xp.Sum(self.r_peak[sid] for sid in self.stations) <= self.params['R_max_peak']
        )
        self.model.addConstraint(
            xp.Sum(self.r_off[sid] for sid in self.stations) <= self.params['R_max_off']
        )

        # Vehicle capacity
        total_rebalancing = xp.Sum(
            self.r_peak[sid] + self.r_off[sid] for sid in self.stations
        )
        self.model.addConstraint(
            total_rebalancing <= self.v * self.params['cap_veh']
        )

    def _add_pricing_constraints(self):
        """Pricing policy"""
        # Peak surcharge
        self.model.addConstraint(
            self.p_peak >= self.p_off * (1 + self.params['delta'])
        )

        # Price elasticity (simplified)
        for od in self.od_pairs:
            self.model.addConstraint(
                self.q_peak[od] <= self.od_pairs_peak[od] * (self.params['p_max'] - self.p_peak) /
                (self.params['p_max'] - self.params['p_min'])
            )

            self.model.addConstraint(
                self.q_off[od] <= self.od_pairs_off[od] * (self.params['p_max'] - self.p_off) /
                (self.params['p_max'] - self.params['p_min'])
            )

    def _add_employer_service_constraints(self):
        """Employer service level constraints"""
        if len(self.employers) == 0:
            print("  WARNING: No employers to constrain!")
            return

        for emp_id, employer in self.employers.items():
            # Count nearby docks
            nearby_docks = xp.Sum(
                self.u[sid] for sid in self.stations
                if self.haversine_distance(
                    employer['lat'], employer['lon'],
                    self.stations[sid]['lat'], self.stations[sid]['lon']
                ) <= self.params['r_walk']
            )

            # Soft constraint (80% of requirement)
            self.model.addConstraint(nearby_docks >= self.params['C_req'] * 0.8)

    def _add_spatial_dispersion_constraint(self):
        """Minimum station spacing"""
        for sid1 in self.stations:
            for sid2 in self.stations:
                if sid1 < sid2:
                    dist = self.station_distances.get((sid1, sid2), float('inf'))
                    if dist < self.params['D_min']:
                        self.model.addConstraint(self.x[sid1] + self.x[sid2] <= 1)

    def _add_school_coverage_constraint(self):
        """School coverage requirements"""
        if len(self.schools) == 0:
            print("  WARNING: No schools to cover!")
            return

        # Link coverage to stations
        for sch_id, school in self.schools.items():
            covered = xp.Sum(
                self.x[sid] for sid in self.stations
                if self.haversine_distance(
                    school['lat'], school['lon'],
                    self.stations[sid]['lat'], self.stations[sid]['lon']
                ) <= self.params['R_cov']
            )

            self.model.addConstraint(self.y_s[sch_id] <= covered)

        # Minimum coverage
        min_schools_covered = int(self.params['gamma'] * len(self.schools))
        self.model.addConstraint(
            xp.Sum(self.y_s[sch_id] for sch_id in self.schools) >= min_schools_covered
        )

    def _is_near_employer(self, sid, emp_id):
        """Check if station is near employer"""
        dist = self.haversine_distance(
            self.employers[emp_id]['lat'], self.employers[emp_id]['lon'],
            self.stations[sid]['lat'], self.stations[sid]['lon']
        )
        return dist <= self.params['r_walk']

    def solve(self, time_limit=300):
        """Solve the model"""
        print("\n  Solving model...")

        self.model.controls.maxtime = time_limit
        self.model.controls.miprelstop = 0.02
        self.model.controls.outputlog = 0

        self.model.solve()

        status = self.model.getProbStatus()
        if status == xp.mip_optimal or status == xp.mip_solution:
            print("  ✓ Solution found!")
            return self._extract_results()
        else:
            print(f"   No solution found (status: {status})")
            return None

    def _extract_results(self):
        """Extract results from optimized model"""
        results = {
            'ridership': 0,
            'ridership_peak': 0,
            'ridership_off': 0,
            'existing_opened': 0,
            'new_opened': 0,
            'total_stations': 0,
            'fleet': self.model.getSolution(self.b) if self.b else 0,
            'vehicles': self.model.getSolution(self.v) if self.v else 0,
            'total_docks': 0,
            'employers_served': 0,
            'schools_covered': 0,
            'avg_station_spacing': 0,
            'employer_satisfaction': 0,
            'employer_capacities': {},
            'fare_peak': self.model.getSolution(self.p_peak) if self.p_peak else 0,
            'fare_off': self.model.getSolution(self.p_off) if self.p_off else 0,
            'rebalancing_peak': 0,
            'rebalancing_off': 0,
            'total_cost': 0,
            'capital_cost': 0,
            'operating_cost': 0,
            'rebalancing_cost': 0,
            'station_locations': {},  # Added for location tracking
            'opened_stations': [],  # List of opened station IDs
        }

        opened_stations = []

        # Extract stations with location information
        for sid in self.stations:
            if self.model.getSolution(self.x[sid]) > 0.5:
                results['total_stations'] += 1
                docks = self.model.getSolution(self.u[sid])
                results['total_docks'] += docks
                opened_stations.append(sid)

                # Store station location information
                results['station_locations'][sid] = {
                    'name': self.stations[sid]['name'],
                    'lat': self.stations[sid]['lat'],
                    'lon': self.stations[sid]['lon'],
                    'capacity': self.stations[sid]['capacity'],
                    'docks_allocated': docks,
                    'station_id': self.stations[sid].get('station_id', ''),
                    'address': self.stations[sid].get('address', ''),
                    'type': 'existing' if sid.startswith('E_') else 'new'
                }

                if sid.startswith('E_'):
                    results['existing_opened'] += 1
                    results['capital_cost'] += self.params['F_existing']
                    results['operating_cost'] += self.params['C_fix_op_existing'] * 365
                else:
                    results['new_opened'] += 1
                    results['capital_cost'] += self.params['F_new']
                    results['operating_cost'] += self.params['C_fix_op_new'] * 365

                results['capital_cost'] += docks * self.params['C_dock']

        # Store list of opened station IDs
        results['opened_stations'] = opened_stations

        results['capital_cost'] += results['fleet'] * self.params['C_bike']

        # Calculate ridership
        for od in self.od_pairs:
            results['ridership_peak'] += self.model.getSolution(self.q_peak[od])
            results['ridership_off'] += self.model.getSolution(self.q_off[od])

        results['ridership'] = results['ridership_peak'] + results['ridership_off']

        # Calculate rebalancing
        for sid in self.stations:
            results['rebalancing_peak'] += self.model.getSolution(self.r_peak[sid])
            results['rebalancing_off'] += self.model.getSolution(self.r_off[sid])

        results['rebalancing_cost'] = (results['rebalancing_peak'] + results['rebalancing_off']) * \
                                      self.params['C_reb'] * 365

        results['total_cost'] = (results['capital_cost'] +
                                 results['operating_cost'] +
                                 results['rebalancing_cost'])

        # Calculate employer metrics
        if len(self.employers) > 0:
            for emp_id, employer in self.employers.items():
                capacity = 0
                for sid in opened_stations:
                    if self._is_near_employer(sid, emp_id):
                        capacity += self.model.getSolution(self.u[sid])

                results['employer_capacities'][emp_id] = capacity

                if capacity >= self.params['C_req']:
                    results['employers_served'] += 1

                # Calculate weighted satisfaction
                results['employer_satisfaction'] += self.employer_weights[emp_id] * capacity

        # Calculate school coverage
        if len(self.schools) > 0:
            for sch_id in self.schools:
                if self.model.getSolution(self.y_s[sch_id]) > 0.5:
                    results['schools_covered'] += 1

        # Average spacing
        if len(opened_stations) > 1:
            spacings = []
            for i, sid1 in enumerate(opened_stations):
                for j, sid2 in enumerate(opened_stations):
                    if i < j:
                        dist = self.station_distances.get((sid1, sid2), 0)
                        if dist > 0:
                            spacings.append(dist)
            if spacings:
                results['avg_station_spacing'] = np.mean(spacings)

        return results

    def run_stochastic_analysis(self, base_results, num_scenarios=100):
        """Stochastic analysis"""
        print("\n  Running stochastic analysis...")

        ridership_samples = []
        satisfaction_samples = []

        for _ in range(num_scenarios):
            # Demand uncertainty
            demand_mult = np.random.normal(1.0, self.params['demand_cv'])

            # Employer participation
            emp_part = np.random.uniform(
                self.params['employer_participation_min'],
                self.params['employer_participation_max']
            )

            ridership_samples.append(base_results['ridership'] * demand_mult)
            satisfaction_samples.append(base_results['employer_satisfaction'] * emp_part)

        return {
            'expected_ridership': np.mean(ridership_samples),
            'ridership_std': np.std(ridership_samples),
            'ridership_95_lower': np.percentile(ridership_samples, 2.5),
            'ridership_95_upper': np.percentile(ridership_samples, 97.5),
            'expected_satisfaction': np.mean(satisfaction_samples),
            'satisfaction_std': np.std(satisfaction_samples),
        }

    def export_station_locations(self, results, filename='selected_stations.csv'):
        """Export selected station locations to CSV"""
        if 'station_locations' in results and results['station_locations']:
            station_data = []
            for sid, info in results['station_locations'].items():
                station_data.append({
                    'station_internal_id': sid,
                    'original_station_id': info.get('station_id', ''),
                    'name': info['name'],
                    'address': info.get('address', ''),
                    'latitude': info['lat'],
                    'longitude': info['lon'],
                    'capacity': info['capacity'],
                    'docks_allocated': info['docks_allocated'],
                    'type': info['type']
                })

            df = pd.DataFrame(station_data)
            df.to_csv(filename, index=False)
            print(f"  → Station locations exported to {filename}")
            return df
        else:
            print("  → No station location data to export")
            return None


def run_comprehensive_analysis():
    """Run complete analysis over multiple budget scenarios"""
    print("=" * 80)
    print("EDINBURGH CYCLE SCHEME OPTIMIZATION ANALYSIS")
    print("=" * 80)

    model = EdinburghCycleOptimizationModel()
    model.load_data()

    # Define scenarios
    council_budgets = [1000000, 1500000, 2000000, 2500000, 3000000]
    cofunding_rates = [0.0, 0.25, 0.5]

    all_results = []
    scenario_count = 0
    total_scenarios = len(council_budgets) * len(cofunding_rates)

    for council_budget in council_budgets:
        for cofunding_rate in cofunding_rates:
            scenario_count += 1
            employer_cofunding = council_budget * cofunding_rate

            print(f"\n{'=' * 80}")
            print(
                f"SCENARIO {scenario_count}/{total_scenarios}: Council £{council_budget / 1e6:.1f}M + {cofunding_rate * 100:.0f}% Co-funding")
            print(f"Total Effective Budget: £{(council_budget + employer_cofunding) / 1e6:.2f}M")
            print(f"{'=' * 80}")

            # STAGE 1: Maximize Ridership
            model.build_model_stage1(council_budget, employer_cofunding)
            stage1_result = model.solve(time_limit=300)

            if stage1_result:
                stage1_result['stage'] = 1
                stage1_result['scenario_id'] = scenario_count
                stage1_result['council_budget'] = council_budget
                stage1_result['cofunding_rate'] = cofunding_rate
                stage1_result['employer_cofunding'] = employer_cofunding
                stage1_result['total_budget'] = council_budget + employer_cofunding

                model.stage1_ridership = stage1_result['ridership']
                model.stage1_solution = stage1_result

                # Stochastic analysis
                stochastic_results = model.run_stochastic_analysis(stage1_result)
                stage1_result.update(stochastic_results)

                print(f"\n  STAGE 1 Results (RIDERSHIP FOCUS):")
                print(f"    Total ridership: {stage1_result['ridership']:,.0f} trips/day")
                print(
                    f"      Peak: {stage1_result['ridership_peak']:,.0f} ({stage1_result['ridership_peak'] / max(1, stage1_result['ridership']) * 100:.1f}%)")
                print(f"      Off-peak: {stage1_result['ridership_off']:,.0f}")
                print(f"    Network: {stage1_result['total_stations']} stations")
                print(f"    Fleet: {stage1_result['fleet']:.0f} bikes, {stage1_result['vehicles']:.0f} vehicles")
                print(f"    Employer Satisfaction: {stage1_result['employer_satisfaction']:.2f}")
                print(f"    Employers Served: {stage1_result['employers_served']}/{len(model.employers)}")
                print(f"    Schools Covered: {stage1_result['schools_covered']}/{len(model.schools)}")
                print(f"    Pricing: Peak £{stage1_result['fare_peak']:.2f}, Off-peak £{stage1_result['fare_off']:.2f}")

                # Export station locations for this scenario
                station_file = f"scenario_{scenario_count}_stage1_stations.csv"
                model.export_station_locations(stage1_result, station_file)

                # STAGE 2: Maximize Employer Satisfaction
                model.build_model_stage2(council_budget, employer_cofunding)
                stage2_result = model.solve(time_limit=300)

                if stage2_result:
                    stage2_result['stage'] = 2
                    stage2_result['scenario_id'] = scenario_count
                    stage2_result['council_budget'] = council_budget
                    stage2_result['cofunding_rate'] = cofunding_rate
                    stage2_result['employer_cofunding'] = employer_cofunding
                    stage2_result['total_budget'] = council_budget + employer_cofunding

                    stochastic_results = model.run_stochastic_analysis(stage2_result)
                    stage2_result.update(stochastic_results)

                    print(f"\n  STAGE 2 Results (EMPLOYER FOCUS):")
                    print(f"    Total ridership: {stage2_result['ridership']:,.0f} trips/day")
                    print(
                        f"    Ridership preserved: {stage2_result['ridership'] / max(1, stage1_result['ridership']) * 100:.1f}%")
                    print(f"    Employer Satisfaction: {stage2_result['employer_satisfaction']:.2f}")

                    if stage1_result['employer_satisfaction'] > 0:
                        improvement = stage2_result['employer_satisfaction'] - stage1_result['employer_satisfaction']
                        pct_improvement = (improvement / stage1_result['employer_satisfaction']) * 100
                        print(f"    Improvement: +{improvement:.2f} ({pct_improvement:.1f}%)")
                    else:
                        print(f"    Improvement: {stage2_result['employer_satisfaction']:.2f} (from 0)")

                    print(f"    Employers Served: {stage2_result['employers_served']}/{len(model.employers)}")

                    # Export station locations for Stage 2
                    station_file = f"scenario_{scenario_count}_stage2_stations.csv"
                    model.export_station_locations(stage2_result, station_file)

                    all_results.append(stage1_result)
                    all_results.append(stage2_result)
                else:
                    print("   Stage 2 optimization failed")
                    all_results.append(stage1_result)
            else:
                print("   Stage 1 optimization failed")

    return all_results, model


if __name__ == "__main__":

    print("\nStarting Edinburgh Cycle Scheme Optimization")

    results, model = run_comprehensive_analysis()

    if results:
        df = pd.DataFrame(results)
        df.to_csv('optimization_results.csv', index=False)
        print(f"\n Results saved to: optimization_results.csv")
        print(f" Analysis complete with {len(model.employers)} employers and {len(model.schools)} schools")
    else:
        print("\n Analysis failed")