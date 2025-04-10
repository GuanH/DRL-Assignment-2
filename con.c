#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct{
    int board[19][19];
    int turn;
}Env;

void reset_env(Env* self){
    memset(self->board, 0, sizeof(self->board));
    self->turn = 1;
}

Env* create_env(){
    Env* self = malloc(sizeof(Env));
    reset_env(self);
    return self;
}

int check_win_env(Env* self){
    int dir[4][2] = {{0,1},{1,0},{1,1},{1,-1}};
    for(int r=0;r<19;r++){
        for(int c=0;c<19;c++){
            if(self->board[r][c] != 0){
                int cur = self->board[r][c];
                for(int i=0;i<4;i++){
                    int dr=dir[i][0], dc=dir[i][1];
                    int prev_r=r-dr, prev_c=c-dc;
                    if(0<=prev_r&&prev_r<19 && 0<=prev_c&&prev_c<19 && self->board[prev_r][prev_c]==cur) continue;
                    int count = 0;
                    int rr=r, cc=c;
                    while(0<=rr&&rr<19 && 0<=cc&&cc<19 && self->board[rr][cc]==cur){
                        count++;
                        rr+=dr;
                        cc+=dc;
                    }
                    if(count >= 6) return cur;
                }
            }
        }
    }
    return 0;
}

void play_env(Env* self, int x, int y){
    self->board[x][y] = self->turn;
    self->turn = 3 - self->turn;
}

void render_env(Env* self){
    for(int i=0;i<19;i++){
        for(int j=0;j<19;j++){
            switch(self->board[i][j]){
                case 0:printf(".");break;
                case 1:printf("X");break;
                case 2:printf("O");break;
            }
        }
        printf("\n");
    }
}


