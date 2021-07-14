#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <queue>
#include <omp.h>
#include <lemon/list_graph.h>
#include <lemon/matching.h>
#include <fstream>

using namespace std;
using namespace lemon;

const bool DEBUG = false;
const bool DEBUG2 = false;
const bool DEBUG3 = false;
const bool IRVINGDEBUG = false;
const bool NONCOVERING = false;
const bool POPBRUTE = false;
const bool QUESTIONS = false;
const bool DETERMINED = false;

#define all(x) begin(x), end(x)

template <class C>
void Print_vector(const C &Original) {
	    for(const auto &v : Original) {
		cout << v << " ";
		}
	        cout << endl;
}

template <class C>
void Print_vector2(const vector<C> &Original) {
	for(const auto &v : Original) {
		cout << v.first << ' ' << v.second << ", ";
	}
	cout << endl;

}


void Print_Matrix(const vector<vector<int>> &Preference_Matrix, int L) {
	    for(int i = 0; i < L; ++i) {
		for(int j = 0; j < (int)Preference_Matrix[i].size(); ++j) {
			cout << Preference_Matrix[i][j] << " ";            
			}
		cout << endl;
		}
}

struct greater_for_sort
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a > b; }
};

struct greater_for_pairs
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a.first > b.first; }
};

struct greater_for_pairs_second
{
	template<class T>
	bool operator()(T const &a, T const &b) const { return a.first > b.first; }
};

class RoomatesProblemGen {
    public:
	int vertex_numb_;
	vector<vector<int>> adjacency_;
	vector<vector<int>> preference_matrix_;
	vector<vector<int>> ranking_matrix_;
	vector<vector<int>> edge_list_;
	
       
	RoomatesProblemGen(int N):vertex_numb_(N) {
		adjacency_.resize(N);
		for(int i = 0; i < N; ++i) {
			adjacency_[i].resize(N, 0);
		}
		
		ranking_matrix_.resize(N);
		for(int i = 0; i < N; ++i) {
			ranking_matrix_[i].resize(N, -1);
		}

		preference_matrix_.resize(N);
	}
	
	void ClearAdjacency() {
		for(vector<int>& row: adjacency_) {
			for(int &elm : row) {
				elm = 0;
			}
		}
	}
	
	void DegreeAdder(int c) {
	//Checks if every vertex's degree is at least n-c, if not then it adds enough edges.
		std::random_device rd; std::mt19937 gen(rd());
		vector<int> vertex_order(vertex_numb_);
		std::iota(all(vertex_order), 0);
		std::shuffle(all(vertex_order), std::mt19937{std::random_device{}()});
		//cout << "VERTEX ORDER: "; Print_vector(vertex_order);
		for(const auto &vert : vertex_order) {
			//cout << "	BEFORE CHECK :\n";
			int ndeg = std::count(all(adjacency_[vert]), 0)-1;
			//cout << vert << " how many missing : " << ndeg << '\n';
			if(ndeg >= c) {
				vector<int> nneigh(ndeg);
				std::copy_if(all(vertex_order), nneigh.begin(), [this, &vert](int x){return vert!=x && adjacency_[vert][x]==0;});
				std::shuffle(all(nneigh), std::mt19937{std::random_device{}()});
				//cout << "NNEIGH: "; Print_vector(nneigh);
				std::uniform_int_distribution<> dist(ndeg-c+1, ndeg);
				int new_edge = dist(gen);
				//cout << "		CHOICE OF ADDED EDGES NUMB: " << new_edge << '\n';
				for(int i = 0; i < new_edge; ++i) {
					adjacency_[vert][nneigh[i]] = 1;
					adjacency_[nneigh[i]][vert] = 1;
				}
				//cout << "		ADDED NEW EDGES:\n";
			}
		
		}
	
	}

	void ErdosRenyi(double prob = 0.2) {
	//prob is probability that edge not exists.
        	std::random_device rd;
            std::mt19937 gen(rd());
        	std::discrete_distribution<> d({prob, 1-prob});	
        	#pragma omp parallel for
        	for(int i = 0; i < vertex_numb_; ++i) {	
        		for(int j = i+1; j < vertex_numb_; ++j) {
        			 adjacency_[i][j] = d(gen); adjacency_[j][i] = adjacency_[i][j];
        		}
        	}
        }

	bool HHCheckRealization(vector<int> &seq) {
		std::sort(all(seq), greater_for_sort());//  cout << "	TRYING SEQUENCE: ";Print_vector(seq);
		while(seq[0] != 0 && seq[(int)seq.size()-1] >= 0 ) {
			std::transform(seq.begin()+1, seq.begin()+seq[0]+1, seq.begin()+1, [](int x){return x-1;}); 
			seq[0]=0;// cout << "	TRANSFORMING SEQ: ";Print_vector(seq);
			std::sort(all(seq), greater_for_sort());// cout << "	TRYING SEQUENCE: ";Print_vector(seq);
		}

		if(seq[(int)seq.size()-1] < 0) return false;
		return true;
	}

	void HHRealization(vector<pair<int,int>> &degrees) {
		std::sort(all(degrees), greater_for_pairs());// cout << "		SORTED DEGSEQ: "; Print_vector2(degrees);
	     while(degrees[0].first != 0 && degrees[(int)degrees.size()-1].first >= 0 ) {
	     	 std::transform(degrees.begin()+1, degrees.begin()+degrees[0].first+1,
	     	 degrees.begin()+1, [this, &degrees](pair<int,int> x)
	     	 {adjacency_[x.second][degrees[0].second] = 1; adjacency_[degrees[0].second][x.second] = 1; return std::make_pair(x.first-1, x.second);}); 
                 degrees[0].first=0; //cout << "	TRANSFORMING SEQ: "; Print_vector2(degrees);
                 std::sort(all(degrees), greater_for_pairs());// cout << "	SORTED DEGSEQ: "; Print_vector2(degrees);
               }
	}

	void HavelHakimi(int c) {
	//Generate a graph by Havel Hakimi algorithm for which every vertex has at least n-c neighbour
		vector<pair<int,int>> degrees(vertex_numb_);
		std::random_device rd; std::mt19937 gen(rd());
		std::uniform_int_distribution<> dist(vertex_numb_-c, vertex_numb_-1);
		while(true) {
			std::generate(all(degrees), [&dist, &gen, i=0]() mutable {return make_pair(dist(gen), i++);});// Print_vector2(degrees);
			if(std::accumulate(all(degrees), 0, [](int a, pair<int,int> b){return a+b.first;})%2 == 0) {
				vector<int> seq(vertex_numb_);
				std::transform(all(degrees), seq.begin(), [](pair<int,int> p){ return p.first;});// Print_vector(seq);
				if(HHCheckRealization(seq)) {
					//cout << "	IT CAN BE REALIZED.\n";
					HHRealization(degrees);
					break;
				}
			}	
		}
	
	}

	void TreeFromPrufer(vector<int> &pruferseq, vector<int> &degrees) {
		vector<int> vertices(vertex_numb_); std::iota(all(vertices), 0);
		for(const auto &s : pruferseq) {
			for(auto &v : vertices) {
				if(degrees[v] == 1) {
					//cout << "FOUND SMALLEST DEG VERT: " << v << " - " << s << '\n';
					adjacency_[s][v] = 1; adjacency_[v][s] = 1;
					degrees[v] -= 1; degrees[s] -= 1;
					break;
				}
			}
		}
		//cout << "BEFORE FINAL: "; Print_vector(degrees); Print_Matrix(adjacency_, vertex_numb_);
		for(auto &v : vertices) {
			if(degrees[v]) {adjacency_[v][vertex_numb_-1] = 1; adjacency_[vertex_numb_-1][v] = 1; break;}
		}
	}

	void PruferGraphCr() {
	//Create a Prufer code from that a tree 
		int n = vertex_numb_-2;
		vector<int> pruferseq(n);
		std::random_device rd; std::mt19937 gen(rd());
		std::uniform_int_distribution<> dist(0, vertex_numb_-1);
		std::generate(all(pruferseq), [&dist, &gen](){return dist(gen);}); //cout << "PRUFER: ";Print_vector(pruferseq);
		vector<int> degrees(vertex_numb_, 1);
		for(const auto &s : pruferseq) {degrees[s] += 1;} // cout << "DEGREES: "; Print_vector(degrees);
		TreeFromPrufer(pruferseq, degrees);// cout << "SO FAR SO GOOD\n"; Print_Matrix(adjacency_, vertex_numb_);
	}

	void CreateEdgeList()  {
		edge_list_.resize(vertex_numb_);
		for(int i = 0; i < vertex_numb_; ++i) {
			for(int j = 0; j < vertex_numb_; ++j) {
				if(adjacency_[i][j]) edge_list_[i].push_back(j);
			}
		}
	}

	void RandomPreference() {
	//Create a random preference list for every vertex
		vector<int> vertices(vertex_numb_); std::iota(all(vertices), 0);
		std::random_device rd; std::mt19937 gen(rd());
		for(const auto &v : vertices) {
			std::copy_if(all(vertices), std::back_inserter(preference_matrix_[v]), [this, &v](int x){return adjacency_[v][x];});
			std::shuffle(all(preference_matrix_[v]), gen);
		}
	}

	void Fill_vector_for_globally(const vector<vector<int>> &globadjacency , const int &who) {
		//Fill who's preference list from globally
        std::priority_queue<pair<int,int>, std::vector<pair<int,int>>, greater_for_pairs_second> Tempi;
		int Count = 0;
		for(int i = 0; i < vertex_numb_; ++i) {
            if(globadjacency[who][i]) {
                Tempi.push(make_pair(globadjacency[who][i], i));
                ++Count;
            }
        }

		preference_matrix_[who].resize(Count);
        for(int i = 0; i < Count; ++i) {
            preference_matrix_[who][Count-i-1] = Tempi.top().second; Tempi.pop();
        }
	}

	int HowManyEdges() {
		int M = 0;
		for(const auto &u : adjacency_) {
			for(const int &v : u) {
				if(v) ++M;
			}
		}
		return M;
	}

	void GlobalPreference() {
		int M = HowManyEdges();
		vector<int> Numbers(M); std::iota(all(Numbers), 1);
		std::random_device rd; std:mt19937 gen(rd());
		std::shuffle(all(Numbers), gen);
		int which = 0; vector<vector<int>> globadjacency = adjacency_;//(vertex_numb_); for(int i = 0; i < vertex_numb_; ++i) globadjacency[i] = adjacency_[i];
		for(auto &u : globadjacency) {
			for(int &v : u) {
				if(v) {v = Numbers[which]; ++which;}
			}
		}
		
		for(int i = 0; i < vertex_numb_; ++i) {
			Fill_vector_for_globally(globadjacency, i);
		}
	}

	void Fill_vector_for_master(const vector<int> &masterlist,const int &who) {
		for(const int &v : masterlist) {
			if(adjacency_[who][v]) preference_matrix_[who].push_back(v);
		}
	}

	void MasterPreference() {
		vector<int> masterlist(vertex_numb_); std::iota(all(masterlist), 0);
		std::random_device rd; std:mt19937 gen(rd());
		std::shuffle(all(masterlist), gen);
		for(int i = 0; i < vertex_numb_; ++i) {
			Fill_vector_for_master(masterlist, i);
		}
	}

	int LowestCommonAncestor(const vector<int> &Ancestor, int u, int v) {
		int curr_u = u, curr_v = v;
		while(curr_u != curr_v) {
			//cout << "Curr_u : " << curr_u << ", Curr_v : " << curr_v << endl;
			if(curr_u == 0) {
				curr_u = v;
			}
			if(curr_v == 0) {
				curr_v = u;
			}
			curr_u = Ancestor[curr_u];
			curr_v = Ancestor[curr_v];
		}
		return curr_u;
	}

	void NonStableModification(int depth = 0) {
		//Searches for an odd length circle and modifies the preference matrix along this circle
		vector<int> Ancestor(vertex_numb_, -1);
		vector<bool> Parity(vertex_numb_, 0);
		int start = 0;
		Ancestor[start] = start; Parity[start] = true;
		queue<int> Waiting; Waiting.push(start);
		pair<int,int> impor_edge; bool prev_parity = true; int curr_depth = 0;
		while(!Waiting.empty()) {
			int u = Waiting.front();
			Waiting.pop();
			for(auto const &v : edge_list_[u]) {
				if(Ancestor[v] == -1) {
					Ancestor[v] = u; Parity[v] = !Parity[u];
					Waiting.push(v);
				}
				if(curr_depth >= depth && Parity[u] == Parity[v]) {
					impor_edge.first = u; impor_edge.second = v;
					while(!Waiting.empty()) {Waiting.pop();}
					break;
				}
			}
			if(prev_parity != Parity[u]) {
				prev_parity = !prev_parity;
				++curr_depth;
			}
		}
		cout << "DEPTH: " << curr_depth << endl;
		cout << impor_edge.first << ", " << impor_edge.second << '\n';

		//Getting lowest common ancestor
		int low_com_anc = LowestCommonAncestor(Ancestor, impor_edge.first, impor_edge.second); cout << " LOW. COM. ANC.: " << low_com_anc << endl;

		// Getting the circle
			vector<int> circ1; int curr_vert = impor_edge.first;
			while(curr_vert != low_com_anc) {circ1.push_back(curr_vert); curr_vert = Ancestor[curr_vert];}
			circ1.push_back(low_com_anc);
			vector<int> circ2; curr_vert = impor_edge.second;
			while(curr_vert != low_com_anc) {circ2.push_back(curr_vert); curr_vert = Ancestor[curr_vert];}
			std::reverse(all(circ2));
			circ1.insert( circ1.end(), all(circ2));
			Print_vector(circ1);

		// Modifying the preference lists 
			for(int i = 0; i < (int)circ1.size()-1; ++i) {
				auto posi_next = std::find(all(preference_matrix_[circ1[i]]), circ1[i+1]); //Making it first in circ1[i]-s list
				auto posi_curr = std::find(all(preference_matrix_[circ1[i+1]]), circ1[i]); //Making it second in circ1[i+1]-s list
				std::iter_swap(preference_matrix_[circ1[i]].begin(), posi_next);
				std::iter_swap(preference_matrix_[circ1[i+1]].begin()+1, posi_curr);
			}
			// Last but not least modfying by import_edge 
				auto posi_next = std::find(all(preference_matrix_[circ1[ (int)circ1.size()-1 ]]), circ1[0]); //Making it first in circ1[i]-s list
				auto posi_curr = std::find(all(preference_matrix_[circ1[0]]), circ1[ (int)circ1.size()-1] ); //Making it second in circ1[i+1]-s list
				std::iter_swap(preference_matrix_[circ1[ (int)circ1.size()-1 ]].begin(), posi_next);
				std::iter_swap(preference_matrix_[circ1[ 0 ]].begin()+1, posi_curr);		

	}

	void RankingMatrixCreate() {
	//Create Ranking Matrixe from Preference Matrix, the higher the more liked
		vector<int> vertices(vertex_numb_); std::iota(all(vertices), 0);
		for(const auto &v : vertices) {
			int likness = (int)preference_matrix_[v].size()-1;
			for(const auto &ne : preference_matrix_[v]) {
				ranking_matrix_[v][ne] = likness--;
			}
		}
	}

};

class RoomatesProblemSol : public RoomatesProblemGen
{
	public:

		RoomatesProblemSol(int N) : RoomatesProblemGen(N) {}

		//More than one member in reduced preference list
		int Locate_Starting_Point(const vector<int> &Left, const vector<int> &Right) {
			for(int i = 0; i < (int)Left.size(); ++i) {
				if(Right[i]-Left[i] >= 1) {
					return i;
				}
			}
			return -1;
		}

		//Is there a preference list without a member?
		int Check_for_zero(const vector<int> &Left, const vector<int> &Right, const vector<int> &first_zeros) {
			for(int i = 0; i < (int)Left.size(); ++i) {
				if(Right[i]-Left[i] < 0 && !first_zeros[i]) {
					return i;
				}
			}
			return -1;
		}

		//Second member in the reduced preference list
		int Second_Searcher(const vector<int> &Left, const vector<int> &Right, int Roter) {
			int Many = 1;
			int second = preference_matrix_[Roter][Left[Roter]+1];
			while( ranking_matrix_[second][Roter] < ranking_matrix_[second][preference_matrix_[second][Right[second]]]) {
				//cout << "   Second: " << second << "  Roter: "<< Roter << "  Right[second]: " << Right[second]  << endl;
				//cout << "   Rank Roter: " << ranking_matrix_[second][Roter] << endl;
				//cout << "   Rank Right second: " << ranking_matrix_[second][preference_matrix_[second][Right[second]]] << endl;
				++Many;
				second = preference_matrix_[Roter][Left[Roter]+Many];
			}
			return Many+Left[Roter];
		}
		
		void IrvingStable() {
			//First Phase 
				//To Whom did you prop, index of hir in your preference list
				vector<int> proposed(vertex_numb_, 0);
				//From Whom did you get a prop
				vector<int> get_prop(vertex_numb_,-1);
				int curr_prop;
				int nex_beg;
				for(int proposer = 0; proposer < vertex_numb_; ++proposer) {
					curr_prop = proposer;
					while(curr_prop != -1 && proposed[curr_prop] < (int)preference_matrix_[curr_prop].size()) {
						int curr_woman = preference_matrix_[curr_prop][proposed[curr_prop]];
						//Does she have a proposer?
						if( get_prop[curr_woman] == -1) {
							get_prop[curr_woman] = curr_prop;
							curr_prop = -1;
						}
						else{
							//The Current Proposer better than the current for the woman?
							if(ranking_matrix_[curr_woman][get_prop[curr_woman]] < ranking_matrix_[curr_woman][curr_prop]) {
									nex_beg = get_prop[curr_woman];
									get_prop[curr_woman] = curr_prop;
									curr_prop = nex_beg;
							}
							else{
									++proposed[curr_prop];
							}
						}
						//cout << get_prop[curr_woman] << '\n';
					}
				}


					cout << "What level is their prop (the indices in the preference matrix where the proposed is): \n";
					Print_vector(proposed);
					cout << "Whom proposed to who: \n";
					Print_vector(get_prop);

				//Left and Right members of int the Reduced Preference Matrix
				//,but Left is Proposed vector and Right is almost Get_Prop

				vector<int> right(vertex_numb_, vertex_numb_-1);
				for(int i = 0; i < vertex_numb_; ++i) {
					if(get_prop[i] != -1) {
						right[i] = (int)preference_matrix_[i].size()-1-ranking_matrix_[i][get_prop[i]];
					}
					else{right[i] = (int)preference_matrix_[i].size()-1;}
				}
				cout << "Right vector: \n";
				Print_vector(right);

				//Search for vertices with already empty preference list and mark it so it won't cause further trouble
				vector<int> first_zeros(vertex_numb_, 0);
				for(int i = 0; i < (int)proposed.size(); ++i) {
					if(right[i]-proposed[i] < 0) {
						first_zeros[i] = 1;
					}
				}


			//Second Phase
				int Roter = Locate_Starting_Point(proposed, right);
				int Zerom = Check_for_zero(proposed, right, first_zeros);
				int qsecond;
				vector<int> Appear(vertex_numb_,0);
				vector<int> Second(vertex_numb_, -1); //int temi = 0;
				while(Roter != -1 && Zerom == -1 ) {
					//++temi;
					//locate an all-or-nothing cycle
						int counter = 1;
						while (Appear[Roter] == 0) {
							Appear[Roter] = counter;
							++counter;
							//Warning Big Warning Second[Roter] gives the index of the second element in the preference list of Roter
							Second[Roter] = Second_Searcher(proposed, right, Roter);
							//cout << "Roter: " << Roter << endl << "Second: " << Second[Roter] << endl;
							Roter = preference_matrix_[preference_matrix_[Roter][Second[Roter]]]
													[right[preference_matrix_[Roter][Second[Roter]]]];
						}

					//cout << "Appear vector" << endl;
					//Print_vector(Appear);

					//Securin the Tail
						vector<int> Tail(counter-Appear[Roter]);
						for(int i = 0; i < vertex_numb_; ++i) {
							if( Appear[i] >= Appear[Roter]) {
								Tail[Appear[i]-Appear[Roter]] = i;
							}
						}
						//cout << "Tail vector" << endl;
						//Print_vector(Tail);           

					//Carry Out a phase 2 reduction
					int Wom;
					for(int i = 0; i < (int)Tail.size(); ++i) {
						proposed[Tail[i]] = Second[Tail[i]];
						Wom = preference_matrix_[Tail[i]][Second[Tail[i]]];
						//cout << "   Tail[i]: " << Tail[i] << endl;
						//cout << "   Wom: " << Wom << endl;
						right[Wom] = (int)preference_matrix_[Wom].size()-1-ranking_matrix_[Wom][Tail[i]];
					}
					//Reinitialization and new values (Starting point etc..)
						std::fill(all(Appear), 0);
						//memset(&Appear[0], 0, Appear.size() * sizeof Appear[0]);
						Roter = Locate_Starting_Point(proposed, right);
						Zerom = Check_for_zero(proposed, right, first_zeros);

					//cout << "Right" << endl;
					//Print_vector(Right);
				}

				if(Zerom == -1) {
					cout << "There is a stable matching and it is: \n";
					for(int i = 0; i < vertex_numb_; ++i) {
						if(proposed[i] < (int)preference_matrix_[i].size())
						cout << i << ": " << preference_matrix_[i][proposed[i]] << endl;
						else{ cout << i << ": " << -1 << '\n';}
					}
				}
				else{
					cout << "There is not any stable matching. \n";
				}
					
		}

		bool IrvingStableBool() {
			//First Phase 
				//To Whom did you prop, index of hir in your preference list
				vector<int> proposed(vertex_numb_, 0);
				//From Whom did you get a prop
				vector<int> get_prop(vertex_numb_,-1);
				int curr_prop;
				int nex_beg;
				for(int proposer = 0; proposer < vertex_numb_; ++proposer) {
					curr_prop = proposer;
					while(curr_prop != -1 && proposed[curr_prop] < (int)preference_matrix_[curr_prop].size()) {
						int curr_woman = preference_matrix_[curr_prop][proposed[curr_prop]];
						//Does she have a proposer?
						if( get_prop[curr_woman] == -1) {
							get_prop[curr_woman] = curr_prop;
							curr_prop = -1;
						}
						else{
							//The Current Proposer better than the current for the woman?
							if(ranking_matrix_[curr_woman][get_prop[curr_woman]] < ranking_matrix_[curr_woman][curr_prop]) {
									nex_beg = get_prop[curr_woman];
									get_prop[curr_woman] = curr_prop;
									curr_prop = nex_beg;
							}
							else{
									++proposed[curr_prop];
							}
						}
						//cout << get_prop[curr_woman] << '\n';
					}
				}

					if(IRVINGDEBUG) {
					cout << "What level is their prop (the indices in the preference matrix where the proposed is): \n";
					Print_vector(proposed);
					cout << "Whom proposed to who: \n";
					Print_vector(get_prop); }

				//Left and Right members of int the Reduced Preference Matrix
				//,but Left is Proposed vector and Right is almost Get_Prop

				vector<int> right(vertex_numb_, vertex_numb_-1);
				for(int i = 0; i < vertex_numb_; ++i) {
					if(get_prop[i] != -1) {
						right[i] = (int)preference_matrix_[i].size()-1-ranking_matrix_[i][get_prop[i]];
					}
					else{right[i] = (int)preference_matrix_[i].size()-1;}
				}
				if(IRVINGDEBUG) {
				cout << "Right vector: \n";
				Print_vector(right); }

				//Search for vertices with already empty preference list and mark it so it won't cause further trouble
				vector<int> first_zeros(vertex_numb_, 0);
				for(int i = 0; i < (int)proposed.size(); ++i) {
					if(right[i]-proposed[i] < 0) {
						first_zeros[i] = 1;
					}
				}


			//Second Phase
				int Roter = Locate_Starting_Point(proposed, right);
				int Zerom = Check_for_zero(proposed, right, first_zeros);
				int qsecond;
				vector<int> Appear(vertex_numb_,0);
				vector<int> Second(vertex_numb_, -1); //int temi = 0;
				while(Roter != -1 && Zerom == -1 ) {
					//++temi;
					//locate an all-or-nothing cycle
						int counter = 1;
						while (Appear[Roter] == 0) {
							Appear[Roter] = counter;
							++counter;
							//Warning Big Warning Second[Roter] gives the index of the second element in the preference list of Roter
							Second[Roter] = Second_Searcher(proposed, right, Roter);
							//cout << "Roter: " << Roter << endl << "Second: " << Second[Roter] << endl;
							Roter = preference_matrix_[preference_matrix_[Roter][Second[Roter]]]
													[right[preference_matrix_[Roter][Second[Roter]]]];
						}

					//cout << "Appear vector" << endl;
					//Print_vector(Appear);

					//Securin the Tail
						vector<int> Tail(counter-Appear[Roter]);
						for(int i = 0; i < vertex_numb_; ++i) {
							if( Appear[i] >= Appear[Roter]) {
								Tail[Appear[i]-Appear[Roter]] = i;
							}
						}
						//cout << "Tail vector" << endl;
						//Print_vector(Tail);           

					//Carry Out a phase 2 reduction
						int Wom;
						for(int i = 0; i < (int)Tail.size(); ++i) {
							proposed[Tail[i]] = Second[Tail[i]];
							Wom = preference_matrix_[Tail[i]][Second[Tail[i]]];
							//cout << "   Tail[i]: " << Tail[i] << endl;
							//cout << "   Wom: " << Wom << endl;
							right[Wom] = (int)preference_matrix_[Wom].size()-1-ranking_matrix_[Wom][Tail[i]];
						}
					//Reinitialization and new values (Starting point etc..)
						std::fill(all(Appear), 0);
						//memset(&Appear[0], 0, Appear.size() * sizeof Appear[0]);
						Roter = Locate_Starting_Point(proposed, right);
						Zerom = Check_for_zero(proposed, right, first_zeros);

					//cout << "Right" << endl;
					//Print_vector(Right);
				}

				if(Zerom == -1) {
					if(IRVINGDEBUG) cout << "There is a stable matching and it is: \n";
					for(int i = 0; i < vertex_numb_; ++i) {
						if(IRVINGDEBUG) {
						if(proposed[i] < (int)preference_matrix_[i].size())
						cout << i << ": " << preference_matrix_[i][proposed[i]] << endl;
						else{ cout << i << ": " << -1 << '\n';} }
					}
					return true;
				}
				else{
					if(DEBUG) cout << "There is not any stable matching." << endl;
					return false;
				}

					
		}

		bool IrvingStableBoolUCovering(const vector<int> &CHARVPZ) {
			//First Phase 
				//To Whom did you prop, index of hir in your preference list
				vector<int> proposed(vertex_numb_, 0);
				//From Whom did you get a prop
				vector<int> get_prop(vertex_numb_,-1);
				int curr_prop;
				int nex_beg;
				for(int proposer = 0; proposer < vertex_numb_; ++proposer) {
					curr_prop = proposer;
					while(curr_prop != -1 && proposed[curr_prop] < (int)preference_matrix_[curr_prop].size()) {
						int curr_woman = preference_matrix_[curr_prop][proposed[curr_prop]];
						//Does she have a proposer?
						if( get_prop[curr_woman] == -1) {
							get_prop[curr_woman] = curr_prop;
							curr_prop = -1;
						}
						else{
							//The Current Proposer better than the current for the woman?
							if(ranking_matrix_[curr_woman][get_prop[curr_woman]] < ranking_matrix_[curr_woman][curr_prop]) {
									nex_beg = get_prop[curr_woman];
									get_prop[curr_woman] = curr_prop;
									curr_prop = nex_beg;
							}
							else{
									++proposed[curr_prop];
							}
						}
						//cout << get_prop[curr_woman] << '\n';
					}
				}

					if(DEBUG3) {
					cout << "What level is their prop (the indices in the preference matrix where the proposed is): \n";
					Print_vector(proposed);
					cout << "Whom proposed to who: \n";
					Print_vector(get_prop); }

				//Left and Right members of int the Reduced Preference Matrix
				//,but Left is Proposed vector and Right is almost Get_Prop

				vector<int> right(vertex_numb_, vertex_numb_-1);
				for(int i = 0; i < vertex_numb_; ++i) {
					if(get_prop[i] != -1) {
						right[i] = (int)preference_matrix_[i].size()-1-ranking_matrix_[i][get_prop[i]];
					}
					else{right[i] = (int)preference_matrix_[i].size()-1;}
				}
				if(DEBUG3) {
				cout << "Right vector: \n";
				Print_vector(right); }

				//Search for vertices with already empty preference list and mark it so it won't cause further trouble
				vector<int> first_zeros(vertex_numb_, 0);
				for(int i = 0; i < (int)proposed.size(); ++i) {
					if(right[i]-proposed[i] < 0) {
						first_zeros[i] = 1;
					}
				}


			//Second Phase
				int Roter = Locate_Starting_Point(proposed, right);
				int Zerom = Check_for_zero(proposed, right, first_zeros);
				int qsecond;
				vector<int> Appear(vertex_numb_,0);
				vector<int> Second(vertex_numb_, -1); //int temi = 0;
				while(Roter != -1 && Zerom == -1 ) {
					//++temi;
					//locate an all-or-nothing cycle
						int counter = 1;
						while (Appear[Roter] == 0) {
							Appear[Roter] = counter;
							++counter;
							//Warning Big Warning Second[Roter] gives the index of the second element in the preference list of Roter
							Second[Roter] = Second_Searcher(proposed, right, Roter);
							//cout << "Roter: " << Roter << endl << "Second: " << Second[Roter] << endl;
							Roter = preference_matrix_[preference_matrix_[Roter][Second[Roter]]]
													[right[preference_matrix_[Roter][Second[Roter]]]];
						}

					//cout << "Appear vector" << endl;
					//Print_vector(Appear);

					//Securin the Tail
						vector<int> Tail(counter-Appear[Roter]);
						for(int i = 0; i < vertex_numb_; ++i) {
							if( Appear[i] >= Appear[Roter]) {
								Tail[Appear[i]-Appear[Roter]] = i;
							}
						}
						//cout << "Tail vector" << endl;
						//Print_vector(Tail);           

					//Carry Out a phase 2 reduction
						int Wom;
						for(int i = 0; i < (int)Tail.size(); ++i) {
							proposed[Tail[i]] = Second[Tail[i]];
							Wom = preference_matrix_[Tail[i]][Second[Tail[i]]];
							//cout << "   Tail[i]: " << Tail[i] << endl;
							//cout << "   Wom: " << Wom << endl;
							right[Wom] = (int)preference_matrix_[Wom].size()-1-ranking_matrix_[Wom][Tail[i]];
						}
					//Reinitialization and new values (Starting point etc..)
						std::fill(all(Appear), 0);
						//memset(&Appear[0], 0, Appear.size() * sizeof Appear[0]);
						Roter = Locate_Starting_Point(proposed, right);
						Zerom = Check_for_zero(proposed, right, first_zeros);

					//cout << "Right" << endl;
					//Print_vector(Right);
				}

				if(Zerom == -1) {
					if(DEBUG2) cout << "There is a stable matching and it is: \n";
					for(int i = 0; i < vertex_numb_; ++i) {
						if(DEBUG2) {
						if(proposed[i] < (int)preference_matrix_[i].size())
						cout << i << ": " << preference_matrix_[i][proposed[i]] << endl;
						else{ cout << i << ": " << -1 << '\n';} }
					}

					for(int i = 0; i < vertex_numb_; ++i) {
						if(!CHARVPZ[i] && proposed[i] >= (int)preference_matrix_[i].size()) {
							if(DEBUG) cout << "NOT OKAY MATCHING BEACUSE it is not complete on V\\V' \n";
							return false;
						}
					}

					return true;
				}
				else{
					if(DEBUG) cout << "There is not any stable matching. \n";
					return false;
				}

					
		}
	
		bool MinusMinusEdge(int u, int v, vector<int> &Pairs) {
			if(ranking_matrix_[u][v] > ranking_matrix_[u][ Pairs[u] ]) {return false;}
			if(ranking_matrix_[v][u] > ranking_matrix_[v][ Pairs[v] ]) {return false;}
			return true;
		}

		void DetermineDs(vector<bool> &alternating_path, vector<int> &Pairs, vector<int> &Ds, vector<int> &CharDs, vector<bool> &not_in_z, int &who) {
			//In alteranting path I indicate who are in the path already (first the blocking edge) and "who" is the current vertex where i am
			//cout << "WHO: " << who << endl;
			if(!alternating_path[ Pairs[who] ]) {
				if(CharDs[ Pairs[who] ] == 0 && !not_in_z[ Pairs[who] ]) {Ds.push_back(Pairs[who]); CharDs[ Pairs[who] ] = 1;}
				alternating_path[ Pairs[who] ] = true;
				DetermineDs(alternating_path, Pairs, Ds, CharDs, not_in_z, Pairs[who]);
				alternating_path[ Pairs[who] ] = false;
			}
			else{
				for(int &u : edge_list_[who]) {
					if(Pairs[u] != -1 && !alternating_path[ u ] && !MinusMinusEdge(u, who, Pairs)) {
						alternating_path[ u ] = true;
						DetermineDs(alternating_path, Pairs, Ds, CharDs, not_in_z, u);
						alternating_path[ u ] = false;
					}
				}
			}

		}

		void CheckingZCoveringPopsMultiple(const vector<int> &Us, vector<int> &Pairs, const vector<int> &Zs, int &depth, bool &end, vector<int> &FinalPop, vector<bool> &not_in_z, bool write) {
			if(end) return;
			if(depth == (int)Zs.size()) {
				//Do everything else
				//Checking for blocking edge in G'= P_Z \union U, P_Z there needs to be one.
				
				vector<pair<int,int>> blocking_edges;
				bool blockin_exits = false;
				for(int i = 0; i < (int)Zs.size(); ++i) {
					int u = Pairs[Zs[i]];
					for(int j = i+1; j < (int)Zs.size(); ++j) {
						int v = Pairs[Zs[j]];
						if(ranking_matrix_[u][v] > ranking_matrix_[u][Zs[i]] && ranking_matrix_[v][u] > ranking_matrix_[v][Zs[j]]) {//cout << "BLOCKING EDGE: " << u << " -> " << v << endl; 
						blockin_exits = true;  blocking_edges.push_back(make_pair(u, v));}
					}
				}

				if(!blockin_exits) {return;}
				if(write) {cout << "BLOCKING EDGES: "; Print_vector2(blocking_edges);}
				
				//Check wheter P_Z is a popular matching
					//G'= P_Z \union U, P_Z
					vector<int> CHARVPZ(vertex_numb_, 0);
					vector<int> VPZ; for(int i = 0; i < vertex_numb_; ++i) {if(Pairs[i] != -1) {VPZ.push_back(i); CHARVPZ[i] = 1;}}
					for(const int &v : Us) {VPZ.push_back(v); CHARVPZ[v] = 1;}
					if(write) {cout << "VPZ: "; Print_vector(VPZ);}
					ListGraph g;
					std::vector<lemon::ListGraph::Node> nodes; nodes.reserve((int)VPZ.size());
					for(int i = 0; i < 2*(int)VPZ.size(); ++i)
						nodes.push_back(g.addNode());
					//cout << "FIRST SEGMENT\n";

					ListGraph::EdgeMap<double> weight(g);
					for(int i = 0; i < (int)VPZ.size(); ++i) {
						//a_i a'_i edge
						ListGraph::Edge FoundEdge = g.addEdge(nodes[i],nodes[i+(int)VPZ.size()]);
						if(Pairs[VPZ[i]] == -1) weight[FoundEdge] = 0;
						else{weight[FoundEdge] = -1;}
						//if(DEBUG) {cout << "OUTER: " << VPZ[i] << ", " << weight[FoundEdge] << endl;}
						for(int j = i+1; j < (int)VPZ.size(); ++j) {
							if(adjacency_[VPZ[i]][VPZ[j]]) {
								ListGraph::Edge FoundEdgeInner = g.addEdge(nodes[i],nodes[j]); //a_i a_j
								ListGraph::Edge FoundEdgeInner2 = g.addEdge(nodes[i+(int)VPZ.size()],nodes[j+(int)VPZ.size()]); //a'_i a'_j
								double delta1, delta2;
								if(Pairs[VPZ[i]] == -1 || ranking_matrix_[ VPZ[i] ][ VPZ[j] ] > ranking_matrix_[ VPZ[i] ][ Pairs[ VPZ[i] ] ]) delta1 = 0.5;
								else{ delta1 = -0.5;}
								if(Pairs[VPZ[j]] == -1 || ranking_matrix_[ VPZ[j] ][ VPZ[i] ] > ranking_matrix_[ VPZ[j] ][ Pairs[ VPZ[j] ] ]) delta2 = 0.5;
								else{ delta2 = -0.5;}
								if(Pairs[VPZ[i]] == VPZ[j]) {delta1 = 0; delta2 = 0;}
        						weight[FoundEdgeInner] = delta1 + delta2;
								weight[FoundEdgeInner2] = delta1 + delta2;
							}
						}
					}

				//Edmonds
					MaxWeightedPerfectMatching<ListGraph, ListGraph::EdgeMap<double>> test(g, weight);
					test.run();
					if(write) cout << "Matching Value : " << test.matchingWeight() << endl;
					
					if(test.matchingWeight() == 0) {
						bool popularif = true;
						//Check if it is a perfect matching
						lemon::ListGraph::NodeIt nod(g); lemon::ListGraph::NodeIt nod2(g); ++nod2;
						if(test.mate(nod) == test.mate(nod2)) {popularif = false; if(DEBUG) cout << "NOT PERFECT MATCHING: " << g.id(nod) << endl;}
						if(popularif) {if(write) cout << "POPULAR MATCHING IN V' FOUND." << endl;}
						else{return;}
						/*
						for(lemon::ListGraph::NodeIt nod(g); nod!=lemon::INVALID; ++nod) {
							//cout << "MATE: " << g.id(nod) << ", " << g.id(test.mate(nod)) << endl;
							if(test.mate(nod) == lemon::INVALID) {popularif = false; break;}
						}
						if(popularif) {if(DEBUG) cout << "POPULAR MATCHING IN V' FOUND.";}
						else{return;}*/
					}
					else{return;}

				//Finding D 
					vector<int> Ds; vector<int> CharDs(vertex_numb_, 0);
					//if((int)edge_list_[Us[0]].size() != vertex_numb_-3) return;
					for(pair<int,int> &e : blocking_edges) {
						if(DETERMINED) cout << "e: " << e.first << ", " << e.second << endl;
						vector<bool> alternating_path1(vertex_numb_, false);
						alternating_path1[e.first] = true; alternating_path1[e.second] = true;
						DetermineDs(alternating_path1, Pairs, Ds, CharDs, not_in_z, e.first);
						vector<bool> alternating_path2(vertex_numb_, false);
						alternating_path2[e.first] = true; alternating_path2[e.second] = true;
						DetermineDs(alternating_path2, Pairs, Ds, CharDs, not_in_z, e.second);
					}
					
					//Ds = Zs;
					if(write) {cout << "D: "; Print_vector(Ds);}
					if(DEBUG2) cout << "D IS DONE\n";

			//Irving in reducated graph
				RoomatesProblemSol StabFindHelper(vertex_numb_);
				//StabFindHelper.adjacency_ = adjacency_;

				for(int i = 0; i < vertex_numb_; ++i) {
						StabFindHelper.adjacency_[i] = adjacency_[i];
					}

				//Delete every V' edge It is not needed
					for(const int &v : VPZ) {
						for(const int &nv : edge_list_[v]) {
							StabFindHelper.adjacency_[v][nv] = 0;
							StabFindHelper.adjacency_[nv][v] = 0;
						}
					}

				for(const int &v : Us) {
						for(const int &nv : edge_list_[v]) {
							//if(StabFindHelper.adjacency_[nv][v])
								for(const int &y : edge_list_[nv])
									if(ranking_matrix_[nv][v] > ranking_matrix_[nv][y]) {StabFindHelper.adjacency_[nv][y] = 0; StabFindHelper.adjacency_[y][nv] = 0;}
						}
					}

				for(const int &z : Ds) {
						for(const int &nz : edge_list_[z]) {
							if(!CHARVPZ[nz] && ranking_matrix_[z][nz] > ranking_matrix_[z][ Pairs[z] ]) return;
							//if(StabFindHelper.adjacency_[nz][z])
								if(!CHARVPZ[nz])
									for(const int &y : edge_list_[nz]) {
										if(ranking_matrix_[nz][z] > ranking_matrix_[nz][y]) {StabFindHelper.adjacency_[nz][y] = 0; StabFindHelper.adjacency_[y][nz] = 0;}
									}
						}
					}

				vector<bool> charact_Us(vertex_numb_, false); for(const auto &v : Us) {charact_Us[v] = true;}				
				for(const int &v : VPZ) {
					if(CharDs[v] == 0 && !charact_Us[v]) {
						for(const int &nv : edge_list_[v]) {
							//if(StabFindHelper.adjacency_[nv][v])
								if(!CHARVPZ[nv] && ranking_matrix_[v][nv] > ranking_matrix_[v][ Pairs[v] ]) {
									for(const int &y : edge_list_[nv]) {
										if(ranking_matrix_[nv][v] > ranking_matrix_[nv][y]) {StabFindHelper.adjacency_[nv][y] = 0; StabFindHelper.adjacency_[y][nv] = 0;}
									}
								}
						}
					}
				}
					

				//Preference Matrix Creation
					for(int i = 0; i < vertex_numb_; ++i) {
						for(int j = 0; j < (int)preference_matrix_[i].size(); ++j) {
							if(StabFindHelper.adjacency_[i][ preference_matrix_[i][j] ]) {StabFindHelper.preference_matrix_[i].push_back(preference_matrix_[i][j]);}
						}
					}
					if((int)VPZ.size() != vertex_numb_) {
						if(write) {cout << "STABLE FINDER HELPER: \n"; Print_Matrix(StabFindHelper.preference_matrix_, vertex_numb_);}
						StabFindHelper.RankingMatrixCreate();
						if(StabFindHelper.IrvingStableBoolUCovering(CHARVPZ)) {end = true; if(write) {cout << "THERE IS A POPULAR NOT U COVERING  MATCHING\n"; Print_vector(Pairs);}}
					}
					else{
						end = true;
					}
					return;
			}
			if(Pairs[ Zs[depth] ] != -1) { ++depth; CheckingZCoveringPopsMultiple(Us, Pairs, Zs, depth, end, FinalPop, not_in_z, write); --depth;}
			for(const auto &v : edge_list_[ Zs[depth] ]) {
				if(Pairs[ Zs[depth] ] == -1 && Pairs[ v ] == -1) {
					Pairs[v] = Zs[depth];  Pairs[ Zs[depth] ] = v; if(write) cout << Zs[depth] << " -> " << v << endl;
					++depth;
					CheckingZCoveringPopsMultiple(Us, Pairs, Zs, depth, end, FinalPop, not_in_z, write);
					--depth;
					Pairs[v] = -1;  Pairs[ Zs[depth] ] = -1;
				}
			}
		}

		bool MultipleNonUCoverPopular(const vector<int> &Us, bool write) {
			//Finding the popular which doesn't cover U or give a proof there is none This one Works for arbitrary size of U

			// Determining the set of Z
				vector<bool> not_in_z(vertex_numb_, false);
				for(const auto &v : Us) {
					not_in_z[v] = true;
					for(const auto &u : edge_list_[v]) {
						not_in_z[u] = true;
					}
				}

				vector<int> base_z;
				for(int i = 0; i < vertex_numb_; ++i) 
					if(!not_in_z[i])
						base_z.push_back(i);
			if(write) {cout << "Us: "; Print_vector(Us);}
			if(write) {cout << "Zs: "; Print_vector(base_z);}

			//Searching for popular matchings
				vector<int> Pairs(vertex_numb_, -1); vector<int> FinalPop(vertex_numb_, -1);
				int depth = 0; bool end = false;
				CheckingZCoveringPopsMultiple(Us, Pairs, base_z, depth, end, FinalPop, not_in_z, write);
				if(write) {cout << "FINAL POPULAR: "; Print_vector(FinalPop);}
				if(end) return true;
				return false;
		}		

		bool CFiveNonUCovering(bool write) {
			int exatly_five = 0;
			for(const auto &v : edge_list_) 
				if((int)v.size() <= vertex_numb_-5 ) 
					++exatly_five;

			if(exatly_five >= 3) {
				for(int v = 0; v < vertex_numb_; ++v) {
					if((int)edge_list_[v].size() <= vertex_numb_-5) {
						vector<int> anti_neigh_v(4); for(int i = 0; i < vertex_numb_; ++i) if(!adjacency_[v][i]) anti_neigh_v.push_back(i);
						for(int &u : anti_neigh_v) {
							vector<int> anti_neigh_uv; for(int i = 0; i < vertex_numb_; ++i) if(!adjacency_[v][i] && !adjacency_[u][i]) anti_neigh_uv.push_back(i);
							if((int)anti_neigh_uv.size() <= 4) {
								for(int &w : anti_neigh_uv) {
									int anti_neighs = 0; for(int i = 0; i < vertex_numb_; ++i) if(!adjacency_[v][i] && !adjacency_[u][i] && !adjacency_[w][i]) ++anti_neighs;
									const vector<int> UKs{v,u,w};
									if(MultipleNonUCoverPopular(UKs, write)) 
										return true;
								}
							}
						}
					}
				}
			}
			for(int k = 0; k < vertex_numb_; ++k)  {
				const vector<int> UKs{k};
				if((int)edge_list_[ UKs[0] ].size() >= vertex_numb_-6) 
					if(MultipleNonUCoverPopular(UKs, write))   
						return true;
			}
			
			return false;
		}

		void PopularMaximalBF(vector<int> &Pairs, int &depth, bool &end) {
			if(end) return;
			if(depth == (int)Pairs.size()) {
				ListGraph g;
				std::vector<lemon::ListGraph::Node> nodes;
				nodes.reserve(2*vertex_numb_);
				for(int i = 0; i < 2*vertex_numb_; ++i)
					nodes.push_back(g.addNode());
					
				ListGraph::EdgeMap<double> weight(g);
				for(int i = 0; i < vertex_numb_; ++i) {
					//a_i a'_i edge
					ListGraph::Edge FoundEdge = g.addEdge(nodes[i],nodes[i+vertex_numb_]);
					if(Pairs[ i ] == -1) weight[FoundEdge] = 0;
					else{weight[FoundEdge] = -1;}
					//if(DEBUG) {cout << "OUTER: " << VPZ[i] << ", " << weight[FoundEdge] << endl;}
					for(int j = i+1; j < vertex_numb_; ++j) {
						if(adjacency_[ i ][ j ]) {
							ListGraph::Edge FoundEdgeInner = g.addEdge(nodes[i],nodes[j]); //a_i a_j
							ListGraph::Edge FoundEdgeInner2 = g.addEdge(nodes[i+vertex_numb_],nodes[j+vertex_numb_]); //a'_i a'_j
							double delta1, delta2;
							if(Pairs[ i ] == -1 || ranking_matrix_[ i ][ j ] > ranking_matrix_[ i ][ Pairs[ i ] ]) delta1 = 0.5;
							else{ delta1 = -0.5;}
							if(Pairs[ j ] == -1 || ranking_matrix_[ j ][ i ] > ranking_matrix_[ j ][ Pairs[ j ] ]) delta2 = 0.5;
							else{ delta2 = -0.5;}
							if(Pairs[ i ] == j) {delta1 = 0; delta2 = 0;}
							weight[FoundEdgeInner] = delta1 + delta2;
							weight[FoundEdgeInner2] = delta1 + delta2;
						}
					}
				}
				
			//Edmonds
				MaxWeightedPerfectMatching<ListGraph, ListGraph::EdgeMap<double>> test(g, weight);
				test.run();
				if(test.matchingWeight() == 0) {
					bool popularif = true;
					//Check if it is a perfect matching
					lemon::ListGraph::NodeIt nod(g); lemon::ListGraph::NodeIt nod2(g); ++nod2;
						if(test.mate(nod) == test.mate(nod2)) {popularif = false; if(DEBUG) cout << "NOT PERFECT MATCHING: " << g.id(nod) << endl;}
						if(popularif) {end = true; if(DEBUG) cout << "POPULAR MATCHING IN V' FOUND." << endl;}
				}
				return;
			}

			if(Pairs[ depth ] != -1) {++depth; PopularMaximalBF(Pairs, depth, end); --depth;}

			if(Pairs[ depth ] == -1) {
				for(const int &v : edge_list_[depth]) {
					if(depth < v && Pairs[ v ] == -1 ) {
						Pairs[ v ] = depth; Pairs[ depth ] = v; ++depth;
						PopularMaximalBF(Pairs, depth, end);
						--depth; Pairs[ v ] = -1; Pairs[ depth ] = -1;
					}
				}

				++depth; PopularMaximalBF(Pairs, depth, end); --depth; //Not in Pairs
			}
		}

		bool PopularBruteForce() {
			vector<int> Pairs(vertex_numb_, -1);
			int depth = 0; bool end = false;
			PopularMaximalBF(Pairs, depth, end);
			if(end) return true;
			return false;
		}
};



int main() {
	if(QUESTIONS) cout << "Hi, you need to answer a few questions from me by the way you have lunched the constant degree changer.\n ";
    if(QUESTIONS) cout << "First: How many vertices should i use in the beggining?\n";
    int N; cin >> N;
    if(QUESTIONS)  cout << "Second: and at the end?\n";
    int End; cin >> End;
    if(QUESTIONS)  cout << "Third: How large should the steps be?\n";
    int Steper; cin >> Steper;
    if(QUESTIONS)  cout << "Fourth: What type of Adjacency Matrix Generation do you want?\n        1,Erdős Rényi\n       2,Havel Hakimi\n      3,Prufer\n";
    int caser; cin >> caser;
    int constantDegree;
    int prop = -1;
	int cdeg;
	if(QUESTIONS)  cout << "Fifth If needed what is the minimum of degree a vertex must have? (Here it means c and every vertex must have n-c neighbour)\n"; 
	cin >> cdeg;

	int prefmodel;
	if(QUESTIONS)  cout << "Sixth: What type of preference generator do you want to use?\n			1, Random\n			2,Global\n			3,Master\n";
	cin >> prefmodel;

    int testNumber;
    if(QUESTIONS)  cout << "Seventh: How many Test should i run?\n";
    cin >> testNumber;

	int indicatingMoving;
	if(QUESTIONS)  cout << "Eigth: Cyclicly after how many should i indicate?\n";
	cin >> indicatingMoving;

    std::ofstream fout;
    fout.open("Changing_Degree_NotCovering.txt");
    fout << N << " " << End << " " << Steper << endl;
    long int sumofStableMatchs = 0; long int trialsinner = 0; long int trialsoutest = 0; long int trialsdegree = 0; long int messageIndicator = 0; long int pluszotv = 0;
    for(int vertexNumb = N; vertexNumb < End+Steper; vertexNumb += Steper) {
        int i = 0;
        while(trialsdegree < testNumber) {
            RoomatesProblemSol Test(vertexNumb);
			switch(caser) {
                case 1: Test.ErdosRenyi(); break;//Test.DegreeAdder(cdeg); break;
                case 2: Test.HavelHakimi(cdeg); break;
                case 3: Test.PruferGraphCr(); Test.DegreeAdder(cdeg) ; break;
            }
			switch(prefmodel) {
				case 1: Test.RandomPreference(); break;
				case 2: Test.GlobalPreference(); break;
				case 3: Test.MasterPreference(); break;
			}
            Test.RankingMatrixCreate();
            Test.CreateEdgeList();
            
			bool LowestDegrreN3 = false; bool BruteForce = false;
			for(int k = 0; k < vertexNumb; ++k) {
				if((int)Test.edge_list_[ k ].size() == vertexNumb-cdeg) LowestDegrreN3 = true;
			}

			bool min_degree = true;
            for(int k = 0; k < vertexNumb; ++k) {
                if((int)Test.edge_list_[ k ].size() < vertexNumb-cdeg) min_degree = false;
            }

			if(min_degree && LowestDegrreN3) {
				if(!Test.IrvingStableBool()) {
					++trialsinner;
					bool PopExistsU = Test.CFiveNonUCovering(false);
					if(PopExistsU) ++i;
				}
				++trialsdegree;
			}
			++trialsoutest;
			++messageIndicator;
			if(messageIndicator >  indicatingMoving) {++pluszotv;  cout << pluszotv << endl; messageIndicator -=  indicatingMoving;}
        }
		cout << "TRIALS: " << trialsoutest << ", " << trialsdegree << ", " << trialsinner << ", " << i << endl; //How many graphs generated?, How many had minimum degree of N-cdeg?, How many didn't have a stable matching?
        sumofStableMatchs = 0;
		trialsinner = 0; trialsoutest = 0; trialsdegree = 0;
    }

	return 0;
}

