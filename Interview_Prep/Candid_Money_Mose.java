/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package leetcode;

import java.util.Arrays;
import java.util.Scanner;

/**
 *
 * @author Lipi
 */
//take Reference of leetcode_problem : https://leetcode.com/problems/coin-change/ 
public class Candid_Money_Mose {

    public void Solution(int b, int w, int[] a) {
        int remaining_balance = b - w;
        //StringBuilder sb = new StringBuilder();
        StringBuffer str = new StringBuffer();
        str.insert(0, remaining_balance);
        int count = 0;

        if(b<w){
            System.out.println("Not enough balance");
            return;
        }
        for (int i = a.length - 1; i >= 0; i--) {
            int packet = w / a[i];
            w = w % a[i];
            //sb.append(packet + ":" + a[i] + " ");
            str.insert(0, packet + ":" + a[i] + " ");
            count = count + packet;
        }
        if (w > 0) {
            System.out.println("Cannot put into packets");
            return;
        }
       // sb.append(" " + remaining_balance);
       // System.out.println(sb.toString());
        System.out.println(str.toString());
        

    }

    public static void main(String[] args) {
        Candid_Money_Mose cmm = new Candid_Money_Mose();
        //take Input from user
        //B = Balance Available
        //W = Withdraw Amount
        //N = No of packets
        //p = packet Size (ml)

        //If Moose downt have enough balance : print "Not enough balance"
        //If balance cannot be broken into available packet output : Print: "Cannot put into packet"
        int b = 1000;
        int w = 567;
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter your available balance:");
        b = sc.nextInt();
        System.out.print("Enter Withraw amount:");
        w = sc.nextInt();
        System.out.print("Enter the number of packages:");
        int n = sc.nextInt();
        int a[]=new int[n];
        System.out.println("Enter each package values which need to be deduct from withdraw amount:");
        for(int i=0;i<n;i++){
            a[i] = sc.nextInt();
        }
        //int[] a = new int[]{5, 2, 10, 50, 100};
        Arrays.sort(a); // a is now sorted ascending array 
        cmm.Solution(b, w, a);

    }
}
