#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>


typedef struct{
    int board[4][4];
    int score;
    int sim_board[4][4];
    int sim_score;
}Game2048Env;

float urand(){
    return (float)(rand()) / RAND_MAX;
}

typedef struct{
    int data[4];
}Row;

void init_row(Row* self, int a[4]){
    for(int i=0;i<4;i++) self->data[i]=a[i];
}

void add_random_tile_2048(Game2048Env* self){
    int n = 0;
    int empty[16][2];
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            if(!self->board[i][j]){
                empty[n][0] = i;
                empty[n][1] = j;
                n++;
            }
        }
    }
    if(n){
        int i = rand() % n;
        int x = empty[i][0];
        int y = empty[i][1];
        if(urand() < 0.9f){
            self->board[x][y] = 2;
        }
        else{
            self->board[x][y] = 4;
        }
    }
}

void reset_2048(Game2048Env* self){
    memset(self->board, 0, sizeof(self->board));
    self->score = 0;
    srand(time(0));
}

Game2048Env* create_game_2048(){
    Game2048Env* self = malloc(sizeof(Game2048Env));
    reset_2048(self);
    return self;
}

void compress_2048(Game2048Env* self, Row* row){
    int i=0;
    for(int j=0;j<4;j++){
        if(row->data[j]){
            row->data[i] = row->data[j];
            i++;
        }
    }
    for(;i<4;i++) row->data[i] = 0;
}

void merge_2048(Game2048Env* self, Row* row){
    for(int i=0;i<3;i++){
        if(row->data[i]==row->data[i+1]&&row->data[i]){
            row->data[i] *= 2;
            row->data[i+1] = 0;
            self->score += row->data[i];
        }
    }
}

int move_left_2048(Game2048Env* self){
    int moved = 0;
    for(int i=0;i<4;i++){
        Row original_row;
        init_row(&original_row, self->board[i]);
        Row new_row;
        init_row(&new_row, self->board[i]);
        compress_2048(self, &new_row);
        merge_2048(self, &new_row);
        compress_2048(self, &new_row);
        for(int j=0;j<4;j++) self->board[i][j] = new_row.data[j];
        for(int j=0;j<4;j++){
            if(original_row.data[j] != self->board[i][j]) moved = 1;
        }
    }
    return moved;
}

void reverse(int a[4]){
    int b[4];
    for(int i=0;i<4;i++)b[i]=a[i];
    for(int i=0;i<4;i++)a[i]=b[3-i];
}

int move_right_2048(Game2048Env* self){
    int moved = 0;
    for(int i=0;i<4;i++){
        Row original_row;
        init_row(&original_row, self->board[i]);
        Row new_row;
        init_row(&new_row, self->board[i]);
        reverse(new_row.data);
        compress_2048(self, &new_row);
        merge_2048(self, &new_row);
        compress_2048(self, &new_row);
        reverse(new_row.data);
        for(int j=0;j<4;j++) self->board[i][j] = new_row.data[j];
        for(int j=0;j<4;j++){
            if(original_row.data[j] != self->board[i][j]) moved = 1;
        }
    }
    return moved;
}

int move_up_2048(Game2048Env* self){
    int moved = 0;
    int boardT[4][4];
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            boardT[i][j] = self->board[j][i];
        }
    }
    for(int i=0;i<4;i++){
        Row original_row;
        init_row(&original_row, boardT[i]);
        Row new_row;
        init_row(&new_row, boardT[i]);
        compress_2048(self, &new_row);
        merge_2048(self, &new_row);
        compress_2048(self, &new_row);
        for(int j=0;j<4;j++) self->board[j][i] = new_row.data[j];
        for(int j=0;j<4;j++){
            if(original_row.data[j] != self->board[j][i]) moved = 1;
        }
    }
    return moved;
}

int move_down_2048(Game2048Env* self){
    int moved = 0;
    int boardT[4][4];
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            boardT[i][j] = self->board[j][i];
        }
    }
    for(int i=0;i<4;i++){
        Row original_row;
        init_row(&original_row, boardT[i]);
        Row new_row;
        init_row(&new_row, boardT[i]);
        reverse(new_row.data);
        compress_2048(self, &new_row);
        merge_2048(self, &new_row);
        compress_2048(self, &new_row);
        reverse(new_row.data);
        for(int j=0;j<4;j++) self->board[j][i] = new_row.data[j];
        for(int j=0;j<4;j++){
            if(original_row.data[j] != self->board[j][i]) moved = 1;
        }
    }
    return moved;
}

int is_game_over_2048(Game2048Env* self){
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            if(!self->board[i][j]) return 0;
            if(i<3&&self->board[i][j]==self->board[i+1][j]) return 0;
            if(j<3&&self->board[i][j]==self->board[i][j+1]) return 0;
        }
    }
    return 1;
}

int step_2048(Game2048Env* self, int action){
    int moved = 0;
    switch(action){
        case 0: moved = move_up_2048(self);break;
        case 1: moved = move_down_2048(self);break;
        case 2: moved = move_left_2048(self);break;
        case 3: moved = move_right_2048(self);break;
    }
    if(moved){
        add_random_tile_2048(self);
    }
    int done = is_game_over_2048(self);
    return done;
}

int afterstate_2048(Game2048Env* self, int action){
    int moved = 0;
    switch(action){
        case 0: moved = move_up_2048(self);break;
        case 1: moved = move_down_2048(self);break;
        case 2: moved = move_left_2048(self);break;
        case 3: moved = move_right_2048(self);break;
    }
    int done = is_game_over_2048(self);
    return done;
}

int is_move_legal_2048(Game2048Env* self, int action){
    Game2048Env tem;
    for(int i=0;i<4;i++)for(int j=0;j<4;j++) tem.board[i][j] = self->board[i][j];
    step_2048(&tem, action);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            if(self->board[i][j]!=tem.board[i][j]) return 1;
        }
    }
    return 0;
}

void render_2048(Game2048Env* self){
    /*
     *  xxxx xxxx xxxx xxxx
     *  xxxx xxxx xxxx xxxx
     *  xxxx xxxx xxxx xxxx
     *  xxxx xxxx xxxx xxxx
     *
     * */
    for(int i=0;i<4;i++){
        printf("|");
        for(int j=0;j<4;j++){
            switch(self->board[i][j]){
                case 0:    printf("    |");break;
                case 2:    printf("   2|");break;
                case 4:    printf("   4|");break;
                case 8:    printf("   8|");break;
                case 16:   printf("  16|");break;
                case 32:   printf("  32|");break;
                case 64:   printf("  64|");break;
                case 128:  printf(" 128|");break;
                case 256:  printf(" 256|");break;
                case 512:  printf(" 512|");break;
                case 1024: printf("1024|");break;
                case 2048: printf("2048|");break;
            }
        }
        printf("\n");
    }
}
void sim_step_2048(Game2048Env* self, int action){
    for(int i=0;i<4;i++)for(int j=0;j<4;j++) self->sim_board[i][j] = self->board[i][j];
    self->sim_score = self->score;
    step_2048(self, action);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            int tem = self->board[i][j];
            self->board[i][j] = self->sim_board[i][j];
            self->sim_board[i][j] = tem;
        }
    }
    int tem = self->score;
    self->score = self->sim_score;
    self->sim_score = tem;
}

void sim_afterstate_2048(Game2048Env* self, int action){
    for(int i=0;i<4;i++)for(int j=0;j<4;j++) self->sim_board[i][j] = self->board[i][j];
    self->sim_score = self->score;
    afterstate_2048(self, action);
    for(int i=0;i<4;i++){
        for(int j=0;j<4;j++){
            int tem = self->board[i][j];
            self->board[i][j] = self->sim_board[i][j];
            self->sim_board[i][j] = tem;
        }
    }
    int tem = self->score;
    self->score = self->sim_score;
    self->sim_score = tem;
}

void set_board_2048(Game2048Env* self, int board[16]){
    for(int i=0,k=0;i<4;i++){
        for(int j=0;j<4;j++,k++){
            self->board[i][j] = board[k];
        }
    }
}

void set_score_2048(Game2048Env* self, int score){
    self->score = score;
}
