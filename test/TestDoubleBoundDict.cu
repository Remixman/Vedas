#include "../src/EmptyIntervalDict.h"
#include <gtest/gtest.h>

#define LARGE_ID 2e9

std::string testVar = "?x";

void testBound(const EmptyIntervalDict& bd, size_t lb1, size_t ub1, size_t lb2, size_t ub2) {
    VAR_BOUND b = bd.getBound(testVar);
    EXPECT_EQ(std::get<0>(b), lb1);
    EXPECT_EQ(std::get<1>(b), ub1);
    EXPECT_EQ(std::get<2>(b), lb2);
    EXPECT_EQ(std::get<3>(b), ub2);
}

TEST(EmptyIntervalDict, NoOverlap) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 1, 3, 500, 600);
  bd.updateBound(testVar, 4, 5, 300, 499);
  testBound(bd, 0, 0, 0, 0);
}

TEST(EmptyIntervalDict, NoOverlap2) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 5, 8, 500, 600);
  bd.updateBound(testVar, 1, 4, 601, 10000);
  testBound(bd, 0, 0, 0, 0);
}

TEST(EmptyIntervalDict, ExactlyEqual) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 4321, 5432, 100000, 200000);
  bd.updateBound(testVar, 4321, 5432, 100000, 200000);
  testBound(bd, 4321, 5432, 100000, 200000);
}

/**
          |------|             |-----|
 |------|                                |->Inf
*/
TEST(EmptyIntervalDict, F1) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 4444, 5555, 100000, 200000);
  bd.updateBound(testVar, 1111, 2222, LARGE_ID, LARGE_ID);
  testBound(bd, 0, 0, 0, 0);
}

/**
          |------|             |-----|
   |------|                             |->Inf
*/
TEST(EmptyIntervalDict, F2) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 4444, 5555, 100000, 200000);
  bd.updateBound(testVar, 3333, 4444, LARGE_ID, LARGE_ID);
  testBound(bd, 4444, 4444, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
       |------|                         |->Inf
*/
TEST(EmptyIntervalDict, F3) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 4444, 6666, 100000, 200000);
  bd.updateBound(testVar, 3333, 5555, LARGE_ID, LARGE_ID);
  testBound(bd, 4444, 5555, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
     |----------------|                 |->Inf
*/
TEST(EmptyIntervalDict, F4) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 4444, 5555, 100000, 200000);
  bd.updateBound(testVar, 3333, 6666, LARGE_ID, LARGE_ID);
  testBound(bd, 4444, 5555, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
          |------|                      |->Inf
*/
TEST(EmptyIntervalDict, F5) {
  EmptyIntervalDict bd;
  bd.updateBound(testVar, 4444, 5555, 100000, 200000);
  bd.updateBound(testVar, 4444, 5555, LARGE_ID, LARGE_ID);
  testBound(bd, 4444, 5555, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
            |---|                       |->Inf
*/
TEST(EmptyIntervalDict, F6) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 3333, 6666, 100000, 200000);
    bd.updateBound(testVar, 4444, 5555, LARGE_ID, LARGE_ID);
    testBound(bd, 4444, 5555, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
              |                         |->Inf
*/
TEST(EmptyIntervalDict, F7) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 4444, 6666, 100000, 200000);
    bd.updateBound(testVar, 5555, 5555, LARGE_ID, LARGE_ID);
    testBound(bd, 5555, 5555, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
               |------|                 |->Inf
*/
TEST(EmptyIntervalDict, F8) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 3333, 5555, 100000, 200000);
    bd.updateBound(testVar, 4444, 6666, LARGE_ID, LARGE_ID);
    testBound(bd, 4444, 5555, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
                 |------|               |->Inf
*/
TEST(EmptyIntervalDict, F9) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 300, 400, 100000, 200000);
    bd.updateBound(testVar, 400, 408, LARGE_ID, LARGE_ID);
    testBound(bd, 400, 400, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
                   |------|             |->Inf
*/
TEST(EmptyIntervalDict, F10) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 1000, 12000, 100000, 200000);
    bd.updateBound(testVar, 12001, 300000, LARGE_ID, LARGE_ID);
    testBound(bd, 0, 0, 0, 0);
}

/**
          |------|             |-----|
                        |------|        |->Inf
*/
TEST(EmptyIntervalDict, F11) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 10, 50, 2000, 3000);
    bd.updateBound(testVar, 1200, 2000, LARGE_ID, LARGE_ID);
    testBound(bd, 2000, 2000, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
                           |------|      |->Inf
*/
TEST(EmptyIntervalDict, F12) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 10, 50, 2000, 3000);
    bd.updateBound(testVar, 1500, 2500, LARGE_ID, LARGE_ID);
    testBound(bd, 1500, 2000, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |-----|
                               |---|      |->Inf
*/
TEST(EmptyIntervalDict, F13) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 10, 50, 2000, 3000);
    bd.updateBound(testVar, 2000, 2500, LARGE_ID, LARGE_ID);
    testBound(bd, 2000, 2500, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |------|
                                 |--|    |->Inf
*/
TEST(EmptyIntervalDict, F14) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 10, 12, 10000, 40000);
    bd.updateBound(testVar, 20000, 30001, LARGE_ID, LARGE_ID);
    testBound(bd, 20000, 30001, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |------|
                                  |      |->Inf
*/
TEST(EmptyIntervalDict, F15) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 10, 12, 10000, 40000);
    bd.updateBound(testVar, 33333, 33333, LARGE_ID, LARGE_ID);
    testBound(bd, 33333, 33333, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |----|
                              |------|    |->Inf
*/
TEST(EmptyIntervalDict, F16) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 1000, 2000, 6000, 9000);
    bd.updateBound(testVar, 4000, 12000, LARGE_ID, LARGE_ID);
    testBound(bd, 6000, 9000, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |----|
                                  |------|   |->Inf
*/
TEST(EmptyIntervalDict, F17) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 1000, 2000, 6000, 9000);
    bd.updateBound(testVar, 7000, 12000, LARGE_ID, LARGE_ID);
    testBound(bd, 7000, 9000, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |----|
                                    |---|  |->Inf
*/
TEST(EmptyIntervalDict, F18) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 1000, 2000, 3000, 4500);
    bd.updateBound(testVar, 4500, 5678, LARGE_ID, LARGE_ID);
    testBound(bd, 4500, 4500, 0, 0); // FIXME: Definition problem
}

/**
          |------|             |----|
                                      |---|  |->Inf
*/
TEST(EmptyIntervalDict, F19) {
    EmptyIntervalDict bd;
    bd.updateBound(testVar, 1000, 2000, 3000, 4500);
    bd.updateBound(testVar, 4501, 5678, LARGE_ID, LARGE_ID);
    testBound(bd, 0, 0, 0, 0);
}