/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package leetcode;

import java.util.*;

/**
 *
 * @author Lipi
 */

// bitcount() Method: It convert integer into binary and find number of bits in it
// https://www.geeksforgeeks.org/java-integer-bitcount-method/
public class BinaryWatch {
    
     public List<String> readBinaryWatch(int num) {
        List<String> list= new ArrayList<>();
        for(int h=0;h<=12;h++){
          for(int m=0;m<=60;m++){
              if((Integer.bitCount(h)+Integer.bitCount(m))==num){
                  if(m<=9){
                  list.add(h+":0"+m);
                  }else{
                  list.add(h+":"+m);
                  }
              }
          }
        }
        return list;
     }
     
     public static void main(String[] args) {
        BinaryWatch bw = new BinaryWatch();
        List<String> list1 =bw.readBinaryWatch(1);
        list1.stream().forEach((list11) -> {
            System.out.println(list11);
         });
    }
} 

